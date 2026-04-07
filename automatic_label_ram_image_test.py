import argparse
import copy
import os
import sys
from types import SimpleNamespace

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(
    0,
    "/home/zxr/Documents/wjc/HRI/project/gaze_point_select_ws/devel/lib/python3/dist-packages",
)
sys.path.append(os.path.join(path, "GroundingDINO"))

# 修改成你的图片文件夹路径。运行后只需要输入图片文件名或相对路径。
IMAGE_DIR = os.path.abspath(os.path.join(path, "test_images"))
INPUT_IMAGE_SIZE = (640, 480)

import cv2
import GroundingDINO.groundingdino.datasets.transforms as T
import numpy as np
import PIL
import supervision as sv
import torch
import torchvision
import torchvision.transforms as TS
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from ram import inference_ram
from ram.models import ram
from segment_anything import SamPredictor, build_sam


class RamGroundedSamImageTest:
    def __init__(
        self,
        device="cuda",
        box_threshold=0.2,
        text_threshold=0.2,
        iou_threshold=0.5,
        ram_threshold=0.68,
    ):
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
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            iou_threshold=iou_threshold,
            ram_threshold=ram_threshold,
            device=device,
            label_mode="1",
            max_area_percentage=1,
            mask_opacity=0.4,
            human_part=[
                "face",
                "hand",
                "shirt",
                "sweatshirt",
                "man",
                "woman",
                "boy",
                "girl",
                "child",
                "businessman",
                "person",
                "people",
                "adult",
                "kid",
                "student",
                "paper",
            ],
        )

        print("Loading RAM model...")
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

        print("Loading Grounded DINO model...")
        self.grounded_model = self._load_grounded_model(
            self.alg_args.grounded_config_file,
            self.alg_args.grounded_checkpoint,
            device=self.alg_args.device,
        )
        print("Loading SAM model...")
        self.sam_model = SamPredictor(
            build_sam(checkpoint=self.alg_args.sam_checkpoint).to(self.alg_args.device)
        )
        print("All models loaded.")

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
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
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
        masks_info = []
        for mask, pred_phrase in zip(masks, pred_phrases):
            area = np.count_nonzero(mask)
            mask_info = {"segmentation": mask, "area": area, "pred_class": pred_phrase}
            masks_info.append(mask_info)
        sorted_masks_info = sorted(masks_info, key=(lambda x: x["area"]))
        np.random.seed(0)
        for mask_info in sorted_masks_info:
            color_mask = [int(c * 255) for c in np.random.random(3)]
            mask_map[mask_info["segmentation"] == True] = color_mask
        image_viz = cv2.addWeighted(image_viz, 1, mask_map, 0.4, gamma=0)
        return image_viz, sorted_masks_info

    @staticmethod
    def _show_result_image(image_viz):
        window_name = "SoM_image"
        cv2.imshow(window_name, image_viz)
        print("Close SoM_image window or press any key in the window to continue.")
        while True:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                break
            key = cv2.waitKey(50)
            if key != -1:
                break
        cv2.destroyWindow(window_name)

    def infer_image(self, image_path):
        image_raw = cv2.imread(image_path)
        if image_raw is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_raw = cv2.resize(
            image_raw, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
        )

        image_viz = copy.deepcopy(image_raw)
        image_height, image_width, _ = image_viz.shape
        image_cv2_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image_pil = PIL.Image.fromarray(image_cv2_rgb)

        person_bboxes_xyxy_and_ids = []
        face_bboxes_xyxy_and_ids = []

        image_pil_resized = image_pil.resize((384, 384))
        image_to_ram = (
            self.transform(image_pil_resized).unsqueeze(0).to(self.alg_args.device)
        )
        ram_inference_result = inference_ram(image_to_ram, self.ram_model)
        ram_tags = ram_inference_result[0].replace(" |", ",")
        ram_tags = [tag.strip() for tag in ram_tags.split(",") if tag.strip() != ""]
        ram_tags = [
            tag for tag in ram_tags if tag.lower() not in self.alg_args.human_part
        ]
        ram_tags = ", ".join(ram_tags)
        # ram_tags = "car, desk, chair, wall "

        labels = []
        pred_phrases = []
        central_pixel_points = []
        person_track_id_index = []
        track_id_list = []
        for bbox_xyxy_and_id in person_bboxes_xyxy_and_ids:
            track_id_list.append(bbox_xyxy_and_id.track_id)

        if ram_tags != "":
            image_transform_pipeline = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_to_grounded, _ = image_transform_pipeline(image_pil, None)
            boxes_filt, scores, pred_phrases = self._get_grounding_output(
                self.grounded_model,
                image_to_grounded,
                ram_tags,
                self.alg_args.box_threshold,
                self.alg_args.text_threshold,
                device=self.alg_args.device,
            )
            pred_phrases_ = []
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
            person_index_after_insert_in_pred_result = [
                i
                for i in range(
                    len(boxes_filt),
                    len(boxes_filt) + len(person_bboxes_xyxy_and_ids),
                )
            ]
            for index, bbox_xyxy_and_id in enumerate(person_bboxes_xyxy_and_ids):
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
            if len(pred_phrases) != 0:
                nms_idx = (
                    torchvision.ops.nms(boxes_filt, scores, self.alg_args.iou_threshold)
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
                if len(pred_phrases) != 0:
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
                    masks = masks.cpu().squeeze(1).numpy()
                    masks_info = []
                    for mask, pred_phrase in zip(masks, pred_phrases):
                        area = np.count_nonzero(mask)
                        H, W = mask.shape
                        mask_info = {
                            "segmentation": mask,
                            "area": area,
                            "pred_class": pred_phrase,
                            "bbox": [0, 0, 0, 0],
                        }
                        masks_info.append(mask_info)
                    sv_detections = sv.Detections.from_sam(masks_info)
                    sorted_masks_info = sorted(
                        masks_info, key=lambda x: x["area"], reverse=True
                    )
                    masks_info_index = list(range(len(masks_info)))
                    masks_info_area = []
                    for mask_info, mask_info_index in zip(masks_info, masks_info_index):
                        masks_info_area.append([mask_info_index, mask_info["area"]])
                    sorted_result = sorted(
                        masks_info_area, key=lambda x: x[1], reverse=True
                    )
                    sorted_index = [result[0] for result in sorted_result]
                    for i, index in enumerate(person_index_in_nms_result):
                        new_index = sorted_index.index(index)
                        person_track_id_index[i][1] = new_index
                    boxes_filt = boxes_filt[sorted_index]
                    pred_phrases_ = []
                    for sorted_mask_info in sorted_masks_info:
                        pred_phrases_.append(sorted_mask_info["pred_class"])
                    pred_phrases = pred_phrases_
                    image_area = H * W
                    mask_annotator = sv.MaskAnnotator(
                        color_lookup=sv.ColorLookup.INDEX,
                        opacity=self.alg_args.mask_opacity,
                    )
                    image_viz = mask_annotator.annotate(
                        scene=image_viz, detections=sv_detections
                    )
                    label_annotator = sv.LabelAnnotator(
                        color_lookup=sv.ColorLookup.INDEX,
                        text_position=sv.Position.DISTANT_TO_BOUNDARY,
                        text_scale=0.4,
                        text_color=sv.Color.WHITE,
                        color=sv.Color.BLACK,
                        text_thickness=1,
                        text_padding=1,
                        smart_position=True,
                    )
                    labels = [str(i) for i in range(len(sv_detections))]
                    image_viz, _ = label_annotator.annotate(
                        scene=image_viz, detections=sv_detections, labels=labels
                    )

                    for bbox in boxes_filt:
                        x1, y1, x2, y2 = bbox.tolist()
                        X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                        X = X if X <= image_width - 1 else int(image_width - 1)
                        Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                        Y = Y if Y <= image_height - 1 else int(image_height - 1)
                        central_pixel_point = [X, Y]
                        central_pixel_points.append(central_pixel_point)
        else:
            if len(track_id_list) != 0:
                raise Exception("RAM Error!")

        person_detect_result = {}
        if len(person_bboxes_xyxy_and_ids) != 0:
            for person_bbox_xyxy_and_id in person_bboxes_xyxy_and_ids:
                person_detect_result[person_bbox_xyxy_and_id.track_id] = (
                    person_bbox_xyxy_and_id.bbox_xyxy
                )

        face_detect_result = {}
        if len(face_bboxes_xyxy_and_ids) != 0:
            for face_bbox_xyxy_and_id in face_bboxes_xyxy_and_ids:
                face_detect_result[face_bbox_xyxy_and_id.track_id] = (
                    face_bbox_xyxy_and_id.bbox_xyxy
                )

        object_index_pairs = []
        track_id_index_pairs = []
        human_like_object_names = {"man", "woman", "boy", "girl", "child"}
        person_index = [track_id_index[1] for track_id_index in person_track_id_index]
        for label_str in labels:
            label = int(label_str)
            if label not in person_index:
                object_index_pair = SimpleNamespace()
                object_index_pair.index = label
                object_index_pair.object_name = pred_phrases[label].split("(")[0]
                central_pixel_point = central_pixel_points[label]
                if object_index_pair.object_name.lower() in human_like_object_names:
                    x1, y1, x2, y2 = boxes_filt[label].tolist()
                    X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                    X = X if X <= image_width - 1 else int(image_width - 1)
                    Y = int(y1 + 0.1 * (y2 - y1))
                    Y = Y if Y >= 0 else 0
                    Y = Y if Y <= image_height - 1 else int(image_height - 1)
                    central_pixel_point = [X, Y]
                object_index_pair.central_pixel_point = central_pixel_point
                object_index_pairs.append(object_index_pair)
            elif label in person_index:
                track_id = -1
                track_id_index_pair = SimpleNamespace()
                for track_id_index in person_track_id_index:
                    if label == track_id_index[1]:
                        track_id = track_id_index[0]
                        break
                track_id_index_pair.track_id = track_id
                track_id_index_pair.index = label
                central_pixel_point = central_pixel_points[label]
                face_bbox_xyxy = face_detect_result.get(track_id, None)
                if face_bbox_xyxy is not None:
                    x1, y1, x2, y2 = face_bbox_xyxy
                    X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                    X = X if X <= image_width - 1 else int(image_width - 1)
                    Y = int((y1 + y2) / 2) if int((y1 + y2) / 2) >= 0 else 0
                    Y = Y if Y <= image_height - 1 else int(image_height - 1)
                    central_pixel_point = [X, Y]
                else:
                    person_bbox_xyxy = person_detect_result.get(track_id, None)
                    if person_bbox_xyxy is not None:
                        x1, y1, x2, y2 = person_bbox_xyxy
                        X = int((x1 + x2) / 2) if int((x1 + x2) / 2) >= 0 else 0
                        X = X if X <= image_width - 1 else int(image_width - 1)
                        Y = int(y1 + 0.1 * (y2 - y1))
                        Y = Y if Y >= 0 else 0
                        Y = Y if Y <= image_height - 1 else int(image_height - 1)
                        central_pixel_point = [X, Y]
                track_id_index_pair.central_pixel_point = central_pixel_point
                track_id_index_pairs.append(track_id_index_pair)

        index_object_name_str = "object: "
        for object_index_pair in object_index_pairs:
            cv2.circle(
                image_viz,
                object_index_pair.central_pixel_point,
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )
            index_object_name_str += (
                f"{object_index_pair.index}-{object_index_pair.object_name}, "
            )

        index_track_id_str = "person: "
        for track_id_index_pair in track_id_index_pairs:
            cv2.circle(
                image_viz,
                track_id_index_pair.central_pixel_point,
                radius=1,
                color=(0, 255, 0),
                thickness=-1,
            )
            index_track_id_str += (
                f"{track_id_index_pair.index}-{track_id_index_pair.track_id}, "
            )

        print(index_object_name_str)
        self._show_result_image(image_viz)

        return image_viz, index_object_name_str


def resolve_image_path(image_name):
    image_name = image_name.strip()
    if os.path.isabs(image_name):
        return image_name
    return os.path.join(IMAGE_DIR, image_name)


def run_interactive_loop(runner):
    print(f"Image directory: {IMAGE_DIR}")
    if not os.path.isdir(IMAGE_DIR):
        print(
            "Warning: image directory does not exist yet. Update IMAGE_DIR in code first."
        )
    print("Enter an image filename to run inference. Type q to quit.")
    while True:
        try:
            image_name = input("image> ").strip()
        except EOFError:
            print("\nExit.")
            break

        if image_name == "":
            continue
        if image_name.lower() in {"q", "quit", "exit"}:
            print("Exit.")
            break

        image_path = resolve_image_path(image_name)
        if not os.path.isfile(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            runner.infer_image(image_path)
        except Exception as exc:
            print(f"Inference failed for {image_path}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Grounded-Segment-Anything Image Test", add_help=True
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device used for inference"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.2, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.2, help="text threshold"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="iou threshold"
    )
    parser.add_argument(
        "--ram_threshold", type=float, default=0.68, help="ram threshold"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    runner = RamGroundedSamImageTest(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        ram_threshold=args.ram_threshold,
    )
    run_interactive_loop(runner)


if __name__ == "__main__":
    main()
