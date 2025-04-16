import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.system("export HF_ENDPOINT=https://hf-mirror.com")
sys.path.insert(
    0,
    "/home/zxr/Documents/wjc/HRI/project/gaze_point_select_ws/devel/lib/python3/dist-packages",
)
import copy
import math
import time
from types import SimpleNamespace

import cv2
import GroundingDINO.groundingdino.datasets.transforms as T
import PIL
import rospy
import torch
import torchvision
from cv_bridge import CvBridge
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

from face_and_person.msg import FacePersonBboxPerImage
from vlm_inference.msg import GazePointTarget, GazePointTarget_1_0


class SmoothPursuit:
    def __init__(self):
        self.alg_args = SimpleNamespace(
            grounded_config_file=os.path.join(
                os.path.dirname(__file__),
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
            grounded_checkpoint=os.path.join(
                os.path.dirname(__file__), "checkpoints/groundingdino_swint_ogc.pth"
            ),
            bert_model_path=os.path.join(
                os.path.dirname(__file__), "checkpoints/bert-base-uncased"
            ),
            device="cuda",
            iou_threshold=0.5,
            box_threshold=0.2,
            text_threshold=0.2,
            pixel_distance_upperbound=200,
        )
        self.smooth_pursuit_target_name = ""
        self.new_smooth_pursuit_target_name = ""
        self.new_pixel_position_x = -1
        self.new_pixel_position_y = -1
        self.update_target_flag = False
        self.last_pixel_position_x = -1
        self.last_pixel_position_y = -1
        self.person_track_id = -1
        self.new_person_track_id = -1
        # grounded model
        self.grounded_model = self._load_grounded_model(
            self.alg_args.grounded_config_file,
            self.alg_args.grounded_checkpoint,
            device=self.alg_args.device,
        )
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/VLM/gaze_point_target",
            GazePointTarget_1_0,
            self._update_target,
            queue_size=1,
        )
        rospy.Subscriber(
            "/face_and_person/face_person_detect_result",
            FacePersonBboxPerImage,
            self._do_smooth_pursuit,
            queue_size=1,
        )
        self.pub_smooth_pursuit_result = rospy.Publisher(
            "/gaze_dm/gaze_point_target", GazePointTarget, queue_size=1
        )

    def _update_target(self, gaze_point_target):
        if gaze_point_target.target_name != "":
            self.new_smooth_pursuit_target_name = gaze_point_target.target_name
            self.new_pixel_position_x = gaze_point_target.pixel_position_x
            self.new_pixel_position_y = gaze_point_target.pixel_position_y
            self.new_person_track_id = gaze_point_target.track_id
            self.update_target_flag = True
        else:
            self.new_smooth_pursuit_target_name = (
                ""  # 注视点在这一推理周期不再变化了,愣住了
            )

    def _do_smooth_pursuit(self, face_person_info):
        color_image = self.bridge.imgmsg_to_cv2(
            face_person_info.color_image, desired_encoding="bgr8"
        )
        cv2.imshow("color_image", color_image)
        cv2.waitKey(1)
        if self.update_target_flag:
            self.update_target_flag = False
            self.smooth_pursuit_target_name = self.new_smooth_pursuit_target_name
            self.person_track_id = self.new_person_track_id
            smooth_pursuit_target_name = self.smooth_pursuit_target_name
            pixel_position_x = self.new_pixel_position_x
            pixel_position_y = self.new_pixel_position_y
            person_track_id = self.person_track_id
            self._do_smooth_pursuit_implement(
                smooth_pursuit_target_name,
                pixel_position_x,
                pixel_position_y,
                person_track_id,
                face_person_info,
            )
        else:
            smooth_pursuit_target_name = self.smooth_pursuit_target_name
            pixel_position_x = self.last_pixel_position_x
            pixel_position_y = self.last_pixel_position_y
            person_track_id = self.person_track_id
            self._do_smooth_pursuit_implement(
                smooth_pursuit_target_name,
                pixel_position_x,
                pixel_position_y,
                person_track_id,
                face_person_info,
            )

    def _do_smooth_pursuit_implement(
        self,
        smooth_pursuit_target_name,
        pixel_position_x,
        pixel_position_y,
        person_track_id,
        face_person_info,
    ):
        if smooth_pursuit_target_name == "":  # 注视点在这一推理周期不再变化了,愣住了
            time.sleep(0.2)
            self._pub_smooth_pursuit_result(
                target_name="",
                pixel_position_x=0,
                pixel_position_y=0,
                depth_image=face_person_info.depth_image,
            )
        else:
            if smooth_pursuit_target_name == "person":
                # color_image = self.bridge.imgmsg_to_cv2(
                #     face_person_info.color_image, desired_encoding="bgr8"
                # )
                # cv2.imshow("color_image", color_image)
                # cv2.waitKey(1)
                # # 先看看能不能找到脸部中心点
                # find_person_face_flag = False
                # for face_bbox_xyxy_and_id in face_person_info.face_bboxes_xyxy_and_ids:
                #     if face_bbox_xyxy_and_id.track_id == person_track_id:
                #         x1, y1, x2, y2 = face_bbox_xyxy_and_id.bbox_xyxy
                #         pixel_position_x = int((x1 + x2) / 2)
                #         pixel_position_y = int((y1 + y2) / 2)
                #         find_person_face_flag = True
                #         break
                # find_person_bbox_face_flag = False
                # if not find_person_face_flag:
                #     for (
                #         person_bbox_xyxy_and_id
                #     ) in face_person_info.person_bboxes_xyxy_and_ids:
                #         if person_bbox_xyxy_and_id.track_id == person_track_id:
                #             x1, y1, x2, y2 = person_bbox_xyxy_and_id.bbox_xyxy
                #             pixel_position_x = int((x1 + x2) / 2)
                #             pixel_position_y = int(y1 + (y2 - y1) * 0.3)
                #             find_person_bbox_face_flag = True
                #             break
                # if find_person_face_flag or find_person_bbox_face_flag:
                #     self._pub_smooth_pursuit_result(
                #         target_name="person",
                #         pixel_position_x=pixel_position_x,
                #         pixel_position_y=pixel_position_y,
                #         depth_image=face_person_info.depth_image,
                #     )
                # else:  # 画面中没有这个人了
                #     self._pub_smooth_pursuit_result(
                #         target_name="",
                #         pixel_position_x=0,
                #         pixel_position_y=0,
                #         depth_image=face_person_info.depth_image,
                #     )
                time.sleep(0.2)
            else:
                time.sleep(0.2)
                # color_image = self.bridge.imgmsg_to_cv2(
                #     face_person_info.color_image, desired_encoding="bgr8"
                # )
                # image_height, image_width, _ = color_image.shape
                # color_image_cv2_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                # image_pil = PIL.Image.fromarray(color_image_cv2_rgb)
                # image_transform_pipeline = T.Compose(
                #     [
                #         T.RandomResize([800], max_size=1333),
                #         T.ToTensor(),
                #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #     ]
                # )
                # image_to_grounded, _ = image_transform_pipeline(
                #     image_pil, None
                # )  # 3, h, w
                # boxes_filt, scores, pred_phrases = self._get_grounding_output(
                #     self.grounded_model,
                #     image_to_grounded,
                #     smooth_pursuit_target_name,
                #     self.alg_args.box_threshold,
                #     self.alg_args.text_threshold,
                #     device=self.alg_args.device,
                # )  # 推理得到的bbox坐标X,Y可能出现小于0的情况
                # if len(pred_phrases) != 0:
                #     size = image_pil.size
                #     H, W = size[1], size[0]
                #     for i in range(boxes_filt.size(0)):
                #         boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                #         boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                #         boxes_filt[i][2:] += boxes_filt[i][:2]
                #     boxes_filt = boxes_filt.cpu()
                #     # use NMS to handle overlapped boxes
                #     nms_idx = (
                #         torchvision.ops.nms(
                #             boxes_filt, scores, self.alg_args.iou_threshold
                #         )
                #         .numpy()
                #         .tolist()
                #     )
                #     boxes_filt = boxes_filt[nms_idx]
                #     min_pixel_distance = 10000
                #     min_distance_index = None
                #     for index, bbox in enumerate(boxes_filt):
                #         x1, y1, x2, y2 = bbox
                #         X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                #         X = X if X <= image_width - 1 else int(image_width - 1)
                #         Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                #         Y = Y if Y <= image_height - 1 else int(image_height - 1)
                #         central_pixel_point = [X, Y]
                #         distance = math.dist(
                #             central_pixel_point,
                #             (pixel_position_x, pixel_position_y),
                #         )
                #         if distance < min_pixel_distance:
                #             min_pixel_distance = distance
                #             min_distance_index = index
                #     if min_pixel_distance <= self.alg_args.pixel_distance_upperbound:
                #         target_bbox = boxes_filt[min_distance_index]
                #         x1, y1, x2, y2 = target_bbox
                #         X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                #         X = X if X <= image_width - 1 else int(image_width - 1)
                #         Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                #         Y = Y if Y <= image_height - 1 else int(image_height - 1)
                #         pixel_position_x = X
                #         pixel_position_y = Y
                #         # print(f"find {smooth_pursuit_target_name}")
                #         print(f"x: {pixel_position_x}, y: {pixel_position_y}")
                #         self._pub_smooth_pursuit_result(
                #             target_name=smooth_pursuit_target_name,
                #             pixel_position_x=pixel_position_x,
                #             pixel_position_y=pixel_position_y,
                #             depth_image=face_person_info.depth_image,
                #         )
                #     else:
                #         self._pub_smooth_pursuit_result(
                #             target_name="",
                #             pixel_position_x=0,
                #             pixel_position_y=0,
                #             depth_image=face_person_info.depth_image,
                #         )
                # else:  # 画面中没有检测到这个物体
                #     self._pub_smooth_pursuit_result(
                #         target_name="",
                #         pixel_position_x=0,
                #         pixel_position_y=0,
                #         depth_image=face_person_info.depth_image,
                #     )
                # self.last_pixel_position_x = pixel_position_x
                # self.last_pixel_position_y = pixel_position_y

    def _pub_smooth_pursuit_result(
        self, target_name, pixel_position_x, pixel_position_y, depth_image
    ):
        smooth_pursuit_result = GazePointTarget()
        smooth_pursuit_result.target_name = target_name
        smooth_pursuit_result.pixel_position_x = pixel_position_x
        smooth_pursuit_result.pixel_position_y = pixel_position_y
        smooth_pursuit_result.depth_image = depth_image
        self.pub_smooth_pursuit_result.publish(smooth_pursuit_result)

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


def main():
    rospy.init_node("smooth_pursuit")
    SmoothPursuit()
    rospy.spin()


if __name__ == "__main__":
    main()
