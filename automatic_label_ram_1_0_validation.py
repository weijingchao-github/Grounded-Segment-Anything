"""
如果运行程序时发现明明有包但报错，比如
ModuleNotFoundError: No module named 'groundingdino'
则按照官方github仓库再安装一遍几个地方
https://github.com/IDEA-Research/Grounded-Segment-Anything
Install without Docker那里：
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-12.1/
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
然后就可以了
"""

"""
目前的设定是连续图像同一物体的mask颜色可能不一样，根据一个index序列赋予颜色，
画面中物体有变化可能就导致了整体颜色变化，可选的已知可以做成同一类一个颜色，根据
traker结果依据tracker id赋颜色。
连续图像人的SoM label可能不一样，和物体一起编序号，从0到最后，不是按照track_id
来，一方面物体没有做track，如果光人依据track_id而物体不依据，只有物体频繁的变SoM
label，可能让大模型只关注人不关注物体的变化，另一方面就是人SoM label依据track_id
的话编号可能就不完整了，比如物体是0-20，人track_id有47.
"""

"""
后面可能需要做的：如果出现摄像头被遮住了的情况，RAM可能没有结果，Grounding DiNO
可能没有结果，不让程序报错，并且返回原始的SoM图像
"""

"""
如果Grounded DINO检测出了face and person没有检测出的人，那就留着它，留在SoM Image上。
"""

import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.system("export HF_ENDPOINT=https://hf-mirror.com")
sys.path.insert(
    0,
    "/home/zxr/Documents/wjc/HRI/project/gaze_point_select_ws/devel/lib/python3/dist-packages",
)
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import copy
import threading
import time
from types import SimpleNamespace

import cv2

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
import matplotlib.colors as mplc
import numpy as np
import PIL
import rospy
import sensor_msgs.msg._Image

# from supervision import supervision as sv
import supervision as sv
import torch
import torchvision
import torchvision.transforms as TS
from cv_bridge import CvBridge
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from ram import inference_ram

# Recognize Anything Model & Tag2Text
from ram.models import ram

# segment anything
from segment_anything import SamPredictor, build_sam
from sensor_msgs.msg import Image

from face_and_person.msg import FacePersonBboxPerImage
from Grounded_SAM.msg import (
    ObjectIndexPair_validation,
    SoM_validation,
    TrackIDIndexPair_validation,
)


class RamGroundedSam:
    def __init__(self):
        # others
        self.alg_args = SimpleNamespace(
            grounded_config_file=os.path.join(
                os.path.dirname(__file__),
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
            ram_checkpoint=os.path.join(
                os.path.dirname(__file__), "checkpoints/ram_swin_large_14m.pth"
            ),
            grounded_checkpoint=os.path.join(
                os.path.dirname(__file__), "checkpoints/groundingdino_swint_ogc.pth"
            ),
            sam_checkpoint=os.path.join(
                os.path.dirname(__file__), "checkpoints/sam_vit_h_4b8939.pth"
            ),
            bert_model_path=os.path.join(
                os.path.dirname(__file__), "checkpoints/bert-base-uncased"
            ),
            # box_threshold=0.05,
            box_threshold=0.2,
            # text_threshold=0.05,
            text_threshold=0.2,
            iou_threshold=0.5,
            ram_threshold=0.68,
            device="cuda",
            label_mode="1",
            max_area_percentage=1,
            mask_opacity=0.4,
            human_part=["face", "hand", "shirt"],
        )
        self.viz_flag = False
        self.recv_counter = 0
        audio_image_pub_frequency = rospy.get_param("/pub_frequency")
        duration = rospy.get_param("/llm_inferecnce_duration")
        self.per_seq_recv_times = audio_image_pub_frequency * duration
        self.do_inference_flag = False
        self.face_person_detect_result = None
        # model init
        # RAM model
        self.ram_model = ram(
            pretrained=self.alg_args.ram_checkpoint,
            image_size=384,
            text_encoder_type=self.alg_args.bert_model_path,
            vit="swin_l",
            threshold=self.alg_args.ram_threshold,
        )
        self.ram_model.eval()
        self.ram_model.to(self.alg_args.device)
        self.normalize = TS.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = TS.Compose(
            [TS.Resize((384, 384)), TS.ToTensor(), self.normalize]
        )
        # grounded model
        self.grounded_model = self._load_grounded_model(
            self.alg_args.grounded_config_file,
            self.alg_args.grounded_checkpoint,
            device=self.alg_args.device,
        )
        # SAM model
        self.sam_model = SamPredictor(
            build_sam(checkpoint=self.alg_args.sam_checkpoint).to(self.alg_args.device)
        )
        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._inference)
        self.loop_thread.start()
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/face_and_person/face_person_detect_result",
            FacePersonBboxPerImage,
            self._process_recv_msg,
            queue_size=10,
        )
        self.pub_SoM_result = rospy.Publisher(
            "SoM_result", SoM_validation, queue_size=10
        )

    def thread_shutdown(self):
        self.thread_running = False
        self.loop_thread.join()

    def _process_recv_msg(self, face_person_detect_result):
        # print(face_person_detect_result.color_image.header.seq)
        self.recv_counter += 1
        if self.recv_counter == self.per_seq_recv_times:
            self.do_inference_flag = True
            self.face_person_detect_result = face_person_detect_result
            # print()
            # print(self.face_person_detect_result.color_image.header.seq)
            # print()
            self.recv_counter = 0

    def _load_grounded_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        args.bert_base_uncased_path = self.alg_args.bert_model_path
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def _get_grounding_output(
        self, model, image, caption, box_threshold, text_threshold, device="cpu"
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def _draw_masks(self, masks, pred_phrases, image_viz):
        mask_map = np.zeros(image_viz.shape, dtype=np.uint8)
        # count every mask's area
        masks_info = []
        for mask, pred_phrase in zip(masks, pred_phrases):
            area = np.count_nonzero(mask)
            mask_info = {"segmentation": mask, "area": area, "pred_class": pred_phrase}
            masks_info.append(mask_info)
        sorted_masks_info = sorted(masks_info, key=(lambda x: x["area"]))
        np.random.seed(0)
        for mask_info in sorted_masks_info:
            color_mask = [int(c * 255) for c in np.random.random(3)]
            # mask_map[:, :, 0][mask_info["segmentation"] == True] = color_mask[0]
            # mask_map[:, :, 1][mask_info["segmentation"] == True] = color_mask[1]
            # mask_map[:, :, 2][mask_info["segmentation"] == True] = color_mask[2]
            mask_map[mask_info["segmentation"] == True] = color_mask
        # cv2.imshow("mask_map", mask_map)
        image_viz = cv2.addWeighted(image_viz, 1, mask_map, 0.4, gamma=0)
        return image_viz, sorted_masks_info

    def _inference(self):
        while self.thread_running:
            if not self.do_inference_flag:
                time.sleep(0.001)
                continue
            self.do_inference_flag = False

            image_raw = self.bridge.imgmsg_to_cv2(
                self.face_person_detect_result.color_image, desired_encoding="bgr8"
            )  # image color
            image_viz = copy.deepcopy(image_raw)
            image_height, image_width, _ = image_viz.shape
            image_cv2_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            image_pil = PIL.Image.fromarray(image_cv2_rgb)
            # RAM inference
            image_pil_resized = image_pil.resize((384, 384))
            image_to_ram = (
                self.transform(image_pil_resized).unsqueeze(0).to(self.alg_args.device)
            )
            ram_inference_result = inference_ram(image_to_ram, self.ram_model)
            ram_tags = ram_inference_result[0].replace(
                " |", ","
            )  # 人并不单纯的只是person，还有man, woman, businessman什么的

            for human_part in self.alg_args.human_part:
                if (human_part + ", ") in ram_tags:
                    ram_tags = ram_tags.replace(human_part + ", ", "")
                if (", " + human_part) in ram_tags:
                    ram_tags = ram_tags.replace(", " + human_part, "")
                if human_part in ram_tags:
                    ram_tags = ram_tags.replace(human_part, "")

            ram_tags_chinese = ram_inference_result[1].replace(" |", ",")

            track_id_list = []
            for (
                bbox_xyxy_and_id
            ) in (
                self.face_person_detect_result.person_bboxes_xyxy_and_ids
            ):  # 接收到的列表里没有人这个loop也能处理
                track_id_list.append(bbox_xyxy_and_id.track_id)
            labels = []  # 等同于SoM index
            pred_phrases = []
            central_pixel_points = []
            person_track_id_index = []
            bboxes_xyxy = []
            if ram_tags != "":
                # Grounded DINO inference
                image_transform_pipeline = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
                image_to_grounded, _ = image_transform_pipeline(
                    image_pil, None
                )  # 3, h, w
                boxes_filt, scores, pred_phrases = self._get_grounding_output(
                    self.grounded_model,
                    image_to_grounded,
                    ram_tags,
                    self.alg_args.box_threshold,
                    self.alg_args.text_threshold,
                    device=self.alg_args.device,
                )  # 推理得到的bbox坐标X,Y可能出现小于0的情况
                pred_phrases_ = (
                    []
                )  # 只留一个类别名称，不然可能会将多个同义的ram识别出的名称拼在一起，作为bbox检测结果，比如"shirt sweatshirt wear(0.43)"
                for pred_phrase in pred_phrases:
                    if len(pred_phrase.split(" ")) == 1:
                        pred_phrases_.append(pred_phrase.split(" ")[0])
                    else:
                        former_part = pred_phrase.split(" ")[0]
                        latter_part = "(" + pred_phrase.split("(")[1]
                        pred_phrases_.append(former_part + latter_part)
                pred_phrases = pred_phrases_
                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]
                boxes_filt = boxes_filt.cpu()
                ## use NMS to handle overlapped boxes
                # 在这里加入face_and_person中检测得到的bbox
                person_index_after_insert_in_pred_result = [
                    i
                    for i in range(
                        len(boxes_filt),
                        len(boxes_filt)
                        + len(
                            self.face_person_detect_result.person_bboxes_xyxy_and_ids
                        ),
                    )
                ]
                # print(self.face_person_detect_result.person_bboxes_xyxy_and_ids)
                for index, bbox_xyxy_and_id in enumerate(
                    self.face_person_detect_result.person_bboxes_xyxy_and_ids
                ):  # 接收到的列表里没有人这个loop也能处理
                    track_id = bbox_xyxy_and_id.track_id
                    person_track_id_index.append(
                        [track_id, person_index_after_insert_in_pred_result[index]]
                    )
                    person_bbox_tensor = torch.tensor(
                        bbox_xyxy_and_id.bbox_xyxy,
                        dtype=boxes_filt.dtype,
                        device=boxes_filt.device,
                    ).unsqueeze(0)
                    boxes_filt = torch.cat((boxes_filt, person_bbox_tensor), dim=0)
                    scores = torch.cat((scores, torch.tensor([1])), dim=0)
                    pred_phrases.append("person(1.00)")
                # （face and person既没检测到人，Grounded DINO又没有检测到超过threshold的物体）的相反情况
                if len(pred_phrases) != 0:
                    # 这里有个问题，就是检测的bbox会不会相互之间会非最大抑制，这个问题已经写了代码来解决
                    nms_idx = (
                        torchvision.ops.nms(
                            boxes_filt, scores, self.alg_args.iou_threshold
                        )
                        .numpy()
                        .tolist()
                    )
                    nms_idx_copy = copy.deepcopy(nms_idx)
                    for index in person_index_after_insert_in_pred_result:
                        if index not in nms_idx:
                            nms_idx_copy.append(index)
                    nms_idx = sorted(nms_idx_copy)
                    boxes_filt = boxes_filt[nms_idx]
                    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                    person_index_in_nms_result = []
                    if len(person_index_after_insert_in_pred_result) != 0:
                        person_index_in_nms_result = list(range(len(nms_idx)))[
                            -1 * len(person_index_after_insert_in_pred_result) :
                        ]
                    # （face and person既没检测到人，nms后没有超过threshold的物体）的相反情况
                    if len(pred_phrases) != 0:
                        # 这里假设对于给定的一个bbox，SAM总能推理出一个mask，不存在不返回mask的情况
                        # SAM inference
                        self.sam_model.set_image(image_cv2_rgb)
                        transformed_boxes = self.sam_model.transform.apply_boxes_torch(
                            boxes_filt, image_cv2_rgb.shape[:2]
                        ).to(self.alg_args.device)
                        masks, _, _ = self.sam_model.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,
                        )
                        # draw output image
                        masks = masks.cpu().squeeze(1).numpy()
                        masks_info = []
                        boxes_filt = boxes_filt.to("cpu").numpy()
                        for mask, pred_phrase, bbox in zip(
                            masks, pred_phrases, boxes_filt
                        ):
                            area = np.count_nonzero(mask)
                            H, W = mask.shape
                            mask_info = {
                                "segmentation": mask,
                                # "segmentation": mask.reshape((1, H, W)),
                                "area": area,
                                "pred_class": pred_phrase,
                                "bbox": list(bbox),
                            }
                            masks_info.append(mask_info)
                        # 这里会打乱masks_info的顺序，从mask面积大的向mask面积小的排
                        sv_detections = sv.Detections.from_sam(masks_info)
                        sorted_masks_info = sorted(
                            masks_info, key=lambda x: x["area"], reverse=True
                        )

                        for mask_info in sorted_masks_info:
                            x1, y1, x2, y2 = mask_info["bbox"]
                            x1 = max(int(x1), 0)
                            y1 = max(int(y1), 0)
                            x2 = min(int(x2), int(image_width - 1))
                            y2 = min(int(y2), int(image_height - 1))
                            bboxes_xyxy.append([x1, y1, x2, y2])

                        masks_info_index = list(range(len(masks_info)))
                        masks_info_area = []
                        for mask_info, mask_info_index in zip(
                            masks_info, masks_info_index
                        ):
                            masks_info_area.append([mask_info_index, mask_info["area"]])
                        sorted_result = sorted(
                            masks_info_area, key=lambda x: x[1], reverse=True
                        )
                        sorted_index = [result[0] for result in sorted_result]
                        for i, index in enumerate(person_index_in_nms_result):
                            new_index = sorted_index.index(index)
                            person_track_id_index[i][1] = new_index
                        pred_phrases_ = []
                        for sorted_mask_info in sorted_masks_info:
                            pred_phrases_.append(sorted_mask_info["pred_class"])
                        pred_phrases = pred_phrases_
                        image_area = H * W
                        # 如果下面这行进行判断要解注释的话，需要保证person不会因为这个条件筛掉
                        # max_area_mask = (
                        #     sv_detections.area / image_area
                        # ) < self.alg_args.max_area_percentage  # 这一行可以用于帮助排除背景物体，目前max_area_percentage，就是背景物体也要
                        # sv_detections = sv_detections[max_area_mask]
                        ## draw masks
                        # 画mask要先画大mask再画小mask，不然如果大小mask重合的话，比如person和face，大mask会覆盖小mask
                        # image_viz, sorted_masks_info = self._draw_masks(masks, pred_phrases, image_viz)
                        mask_annotator = sv.MaskAnnotator(
                            color_lookup=sv.ColorLookup.INDEX,
                            opacity=self.alg_args.mask_opacity,
                        )
                        image_viz = mask_annotator.annotate(
                            scene=image_viz, detections=sv_detections
                        )
                        ## draw lables/draw index
                        # self._draw_label(sorted_masks_info, self.alg_args.label_mode, image_viz)
                        label_annotator = sv.LabelAnnotator(
                            color_lookup=sv.ColorLookup.INDEX,
                            text_position=sv.Position.DISTANT_TO_BOUNDARY,
                            text_scale=0.4,  # text的大小
                            text_color=sv.Color.WHITE,
                            color=sv.Color.BLACK,
                            text_thickness=1,
                            text_padding=1,  # 文字左边或右边单边padding多少个像素
                            smart_position=True,
                        )
                        # TODO: label type根据场景变化
                        labels = [str(i) for i in range(len(sv_detections))]
                        image_viz, label_pixel_background_position_xyxy_list = (
                            label_annotator.annotate(
                                scene=image_viz, detections=sv_detections, labels=labels
                            )
                        )

                        for (
                            label_pixel_background_position_xyxy
                        ) in label_pixel_background_position_xyxy_list:
                            x1, y1, x2, y2 = label_pixel_background_position_xyxy
                            X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                            X = X if X <= image_width - 1 else int(image_width - 1)
                            Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                            Y = Y if Y <= image_height - 1 else int(image_height - 1)
                            central_pixel_point = [X, Y]
                            central_pixel_points.append(central_pixel_point)
                    else:
                        # nothing needs to handle
                        pass
                else:
                    # nothing needs to handle
                    pass
            else:
                if len(track_id_list) != 0:
                    raise Exception("RAM Error!")

            # pub msg
            SoM_result = SoM_validation()
            SoM_result.seq_id = self.face_person_detect_result.seq_id
            SoM_result.SoM_image = self.bridge.cv2_to_imgmsg(image_viz, encoding="bgr8")
            SoM_result.color_image = self.face_person_detect_result.color_image
            SoM_result.depth_image = self.face_person_detect_result.depth_image

            face_detect_result = {}
            if len(self.face_person_detect_result.person_bboxes_xyxy_and_ids) != 0:
                for (
                    face_bbox_xyxy_and_id
                ) in self.face_person_detect_result.face_bboxes_xyxy_and_ids:
                    face_detect_result[face_bbox_xyxy_and_id.track_id] = (
                        face_bbox_xyxy_and_id.bbox_xyxy
                    )

            SoM_result.object_index_pairs = []
            SoM_result.track_id_index_pairs = []
            person_index = [
                track_id_index[1] for track_id_index in person_track_id_index
            ]
            for label_str in labels:
                label = int(label_str)
                if label not in person_index:
                    object_index_pair = ObjectIndexPair_validation()
                    object_index_pair.index = label
                    object_index_pair.object_name = pred_phrases[label].split("(")[0]
                    object_index_pair.central_pixel_point = central_pixel_points[label]
                    object_index_pair.bbox_xyxy = bboxes_xyxy[label]
                    SoM_result.object_index_pairs.append(object_index_pair)
                elif label in person_index:
                    track_id = -1
                    track_id_index_pair = TrackIDIndexPair_validation()
                    for track_id_index in person_track_id_index:
                        if label == track_id_index[1]:
                            track_id = track_id_index[0]
                            break
                    track_id_index_pair.track_id = track_id
                    track_id_index_pair.index = label
                    # 如果没有检测到这个人的脸，就先设定看向这个人的person bbox中心吧
                    central_pixel_point = central_pixel_points[label]
                    # face_bbox_xyxy = face_detect_result.get(track_id, None)
                    # if face_bbox_xyxy is not None:
                    #     x1, y1, x2, y2 = face_bbox_xyxy
                    #     X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                    #     X = X if X <= image_width - 1 else int(image_width - 1)
                    #     Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                    #     Y = Y if Y <= image_height - 1 else int(image_height - 1)
                    #     central_pixel_point = [X, Y]
                    track_id_index_pair.central_pixel_point = central_pixel_point
                    track_id_index_pair.bbox_xyxy = bboxes_xyxy[label]
                    SoM_result.track_id_index_pairs.append(track_id_index_pair)

            self.pub_SoM_result.publish(SoM_result)

            if self.viz_flag:
                # object
                index_object_name_str = "object: "
                for object_index_pair in SoM_result.object_index_pairs:
                    cv2.circle(
                        image_viz,
                        object_index_pair.central_pixel_point,
                        radius=1,
                        color=(0, 0, 255),
                        thickness=-1,
                    )  # 物体用绿色的点
                    index_object_name_str += (
                        f"{object_index_pair.index}-{object_index_pair.object_name}, "
                    )
                # print(index_object_name_str)
                # person
                index_track_id_str = "person: "
                for track_id_index_pair in SoM_result.track_id_index_pairs:
                    cv2.circle(
                        image_viz,
                        track_id_index_pair.central_pixel_point,
                        radius=1,
                        color=(0, 255, 0),
                        thickness=-1,
                    )  # 人用红色的点
                    index_track_id_str += (
                        f"{track_id_index_pair.index}-{track_id_index_pair.track_id}, "
                    )
                # print(index_track_id_str)
                # SoM image
                cv2.imshow("SoM_image", image_viz)
                cv2.waitKey(1)


def main():
    rospy.init_node("ram_grounded_sam")
    ram_grounded_sam = RamGroundedSam()
    try:
        rospy.spin()
    finally:
        ram_grounded_sam.thread_shutdown()


if __name__ == "__main__":
    main()
