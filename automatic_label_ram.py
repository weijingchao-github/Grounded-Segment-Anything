import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
from types import SimpleNamespace

import cv2

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
import matplotlib.colors as mplc
import numpy as np
import PIL
import rospy

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


class RamGroundedSam:
    def __init__(self):
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
            # box_threshold=0.05,
            box_threshold=0.25,
            # text_threshold=0.05,
            text_threshold=0.2,
            iou_threshold=0.5,
            device="cuda",
            label_mode="1",
            max_area_percentage=1,
            mask_opacity=0.4,
        )
        self.enable_viz = True
        # model init
        # RAM model
        self.ram_model = ram(
            pretrained=self.alg_args.ram_checkpoint, image_size=384, vit="swin_l"
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
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self._inference,
            queue_size=1,
        )

    def _load_grounded_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        # print(load_res)
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

    # def _draw_label(self, sorted_masks_info, label_mode, image_viz):
    #     lighter_color = [1, 1, 1]
    #     label = 1
    #     for mask_info in sorted_masks_info:
    #         binary_mask = mask_info["segmentation"]

    #         def number_to_string(n):
    #             chars = []
    #             while n:
    #                 n, remainder = divmod(n - 1, 26)
    #                 chars.append(chr(97 + remainder))
    #             return "".join(reversed(chars))

    #         binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), "constant")
    #         mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
    #         mask_dt = mask_dt[1:-1, 1:-1]
    #         max_dist = np.max(mask_dt)
    #         coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]

    #         if label_mode == "a":
    #             label_str = number_to_string(label)
    #         else:
    #             label_str = str(label)
    #         H, W, _ = image_viz.shape
    #         scale = 1.0
    #         font_size = max(np.sqrt(H * W) // 90, 10 // scale)
    #         self._draw_text(
    #             label_str,
    #             (coords_x[len(coords_x) // 2] + 2, coords_y[len(coords_y) // 2] - 6),
    #             font_size=font_size,
    #             color=lighter_color,
    #         )

    #         label += 1

    # def _draw_text(
    #     self,
    #     text,
    #     position,
    #     *,
    #     font_size=None,
    #     color="g",
    #     horizontal_alignment="center",
    #     rotation=0,
    # ):
    #     """
    #     Args:
    #         text (str): class label
    #         position (tuple): a tuple of the x and y coordinates to place text on image.
    #         font_size (int, optional): font of the text. If not provided, a font size
    #             proportional to the image width is calculated and used.
    #         color: color of the text. Refer to `matplotlib.colors` for full list
    #             of formats that are accepted.
    #         horizontal_alignment (str): see `matplotlib.text.Text`
    #         rotation: rotation angle in degrees CCW

    #     Returns:
    #         output (VisImage): image object with text drawn.
    #     """

    #     # since the text background is dark, we don't want the text to be dark
    #     color = np.maximum(list(mplc.to_rgb(color)), 0.15)
    #     color[np.argmax(color)] = max(0.8, np.max(color))

    #     def contrasting_color(rgb):
    #         """Returns 'white' or 'black' depending on which color contrasts more with the given RGB value."""

    #         # Decompose the RGB tuple
    #         R, G, B = rgb

    #         # Calculate the Y value
    #         Y = 0.299 * R + 0.587 * G + 0.114 * B

    #         # If Y value is greater than 128, it's closer to white so return black. Otherwise, return white.
    #         return "black" if Y > 128 else "white"

    #     bbox_background = contrasting_color(color * 255)

    #     x, y = position
    #     self.output.ax.text(
    #         x,
    #         y,
    #         text,
    #         size=font_size * self.output.scale,
    #         family="sans-serif",
    #         bbox={
    #             "facecolor": bbox_background,
    #             "alpha": 0.8,
    #             "pad": 0.7,
    #             "edgecolor": "none",
    #         },
    #         verticalalignment="top",
    #         horizontalalignment=horizontal_alignment,
    #         color=color,
    #         zorder=10,
    #         rotation=rotation,
    #     )
    #     return self.output

    def _inference(self, image_rev):
        image_raw = self.bridge.imgmsg_to_cv2(image_rev, desired_encoding="bgr8")
        image_viz = copy.deepcopy(image_raw)
        image_cv2_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image_pil = PIL.Image.fromarray(image_cv2_rgb)
        # RAM inference
        image_pil_resized = image_pil.resize((384, 384))
        image_to_ram = (
            self.transform(image_pil_resized).unsqueeze(0).to(self.alg_args.device)
        )
        ram_inference_result = inference_ram(image_to_ram, self.ram_model)
        ram_tags = ram_inference_result[0].replace(" |", ",")
        ram_tags_chinese = ram_inference_result[1].replace(" |", ",")
        # Grounded DINO inference
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
            ram_tags,
            self.alg_args.box_threshold,
            self.alg_args.text_threshold,
            device=self.alg_args.device,
        )
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        ## use NMS to handle overlapped boxes
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, self.alg_args.iou_threshold)
            .numpy()
            .tolist()
        )
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
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
        for mask, pred_phrase in zip(masks, pred_phrases):
            area = np.count_nonzero(mask)
            H, W = mask.shape
            mask_info = {
                "segmentation": mask,
                # "segmentation": mask.reshape((1, H, W)),
                "area": area,
                "pred_class": pred_phrase,
                "bbox": [0, 0, 0, 0],
            }
            masks_info.append(mask_info)
        sv_detections = sv.Detections.from_sam(masks_info)
        image_area = H * W
        max_area_mask = (
            sv_detections.area / image_area
        ) < self.alg_args.max_area_percentage
        sv_detections = sv_detections[max_area_mask]
        ## draw masks
        # image_viz, sorted_masks_info = self._draw_masks(masks, pred_phrases, image_viz)
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=self.alg_args.mask_opacity
        )
        image_viz = mask_annotator.annotate(scene=image_viz, detections=sv_detections)
        ## draw lables
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
        image_viz = label_annotator.annotate(
            scene=image_viz, detections=sv_detections, labels=labels
        )

        if self.enable_viz:
            cv2.imshow("image", image_viz)
            cv2.waitKey(1)


def main():
    rospy.init_node("automatic_label_ram")
    RamGroundedSam()
    rospy.spin()


if __name__ == "__main__":
    main()
