import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.system("export HF_ENDPOINT=https://hf-mirror.com")

import copy
import math
import threading
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

from vlm_inference.msg import GazePointTarget_1_0_validation


class GetGazePointTargetBbox:
    def __init__(self, f):
        self.f = f
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
            box_threshold=0.25,
            text_threshold=0.2,
            pixel_distance_upperbound=200,
        )
        # grounded model
        self.grounded_model = self._load_grounded_model(
            self.alg_args.grounded_config_file,
            self.alg_args.grounded_checkpoint,
            device=self.alg_args.device,
        )
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/VLM/gaze_point_target",
            GazePointTarget_1_0_validation,
            self._get_bbox,
            queue_size=1,
        )

    def _get_bbox(self, gaze_point_target):
        if gaze_point_target.target_name == "":
            self.f.write("\n")
        else:
            print(gaze_point_target.target_name)
            color_image = self.bridge.imgmsg_to_cv2(
                gaze_point_target.color_image, desired_encoding="bgr8"
            )
            image_height, image_width, _ = color_image.shape
            color_image_cv2_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_pil = PIL.Image.fromarray(color_image_cv2_rgb)
            image_transform_pipeline = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_to_grounded, _ = image_transform_pipeline(image_pil, None)  # 3, h, w
            boxes_filt, scores, pred_phrases = self._get_grounding_output(
                self.grounded_model,
                image_to_grounded,
                gaze_point_target.target_name,
                self.alg_args.box_threshold,
                self.alg_args.text_threshold,
                device=self.alg_args.device,
            )  # 推理得到的bbox坐标X,Y可能出现小于0的情况
            if len(pred_phrases) != 0:
                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]
                boxes_filt = boxes_filt.cpu()
                # use NMS to handle overlapped boxes
                nms_idx = (
                    torchvision.ops.nms(boxes_filt, scores, self.alg_args.iou_threshold)
                    .numpy()
                    .tolist()
                )
                boxes_filt = boxes_filt[nms_idx]
                min_pixel_distance = 10000
                min_distance_index = None
                pixel_position_x = gaze_point_target.pixel_position_x
                pixel_position_y = gaze_point_target.pixel_position_y
                for index, bbox in enumerate(boxes_filt):
                    x1, y1, x2, y2 = bbox
                    x1 = max(int(x1), 0)
                    y1 = max(int(y1), 0)
                    x2 = min(int(x2), int(image_width - 1))
                    y2 = min(int(y2), int(image_height - 1))
                    X = int((x1 + x2) / 2)
                    Y = int((y1 + y2) / 2)
                    central_pixel_point = [X, Y]
                    distance = math.dist(
                        central_pixel_point,
                        (pixel_position_x, pixel_position_y),
                    )
                    if distance < min_pixel_distance:
                        min_pixel_distance = distance
                        min_distance_index = index

                if min_pixel_distance <= self.alg_args.pixel_distance_upperbound:
                    target_bbox = boxes_filt[min_distance_index]
                    x1, y1, x2, y2 = target_bbox
                    x1 = max(int(x1), 0)
                    y1 = max(int(y1), 0)
                    x2 = min(int(x2), int(image_width - 1))
                    y2 = min(int(y2), int(image_height - 1))
                    self.f.write(f"{x1} {y1} {x2} {y2}\n")
                    cv2.rectangle(
                        color_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        thickness=3,
                    )
                    cv2.imshow("color_image", color_image)
                    cv2.waitKey(1)
                else:
                    self.f.write("Grounding model detect result far from gps target.\n")
            else:
                self.f.write("Return no bbox from grounding model.\n")

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
    rospy.init_node("get_gaze_point_object_bbox")
    data_folder_name = rospy.get_param("/data_folder_name")
    video_name = rospy.get_param("/video_name")
    video_name = video_name.split(".mp4")[0]
    current_file_path = os.path.abspath(__file__)
    save_path = os.path.join(
        current_file_path.split("/src/")[0]  # ROS workspace path
        + "/experiment/"
        + data_folder_name
        + "/save_result",
        video_name,
    )
    with open(os.path.join(save_path, "gaze_point_target_bbox.txt"), "w") as f:
        GetGazePointTargetBbox(f)
        rospy.spin()


if __name__ == "__main__":
    main()
