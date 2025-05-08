import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.system("export HF_ENDPOINT=https://hf-mirror.com")

from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import torch
from segment_anything import SamPredictor, build_sam


class AddMask:
    def __init__(self):
        self.viz_flag = True
        self.alg_args = SimpleNamespace(
            sam_checkpoint=os.path.join(
                os.path.dirname(__file__), "checkpoints/sam_vit_h_4b8939.pth"
            ),
            device="cuda",
            mask_opacity=0.5,
        )
        # load SAM model
        self.sam_model = SamPredictor(
            build_sam(checkpoint=self.alg_args.sam_checkpoint).to(self.alg_args.device)
        )
        # add mask
        data_folder_name = rospy.get_param("/data_folder_name")
        video_name = rospy.get_param("/video_name")
        times = rospy.get_param("/times")
        video_name = video_name.split(".mp4")[0]
        current_file_path = os.path.abspath(__file__)
        save_path = os.path.join(
            current_file_path.split("/src/")[0]  # ROS workspace path
            + "/experiment/"
            + data_folder_name
            + "/save_result",
            video_name + f"/{times}",
        )
        vlm_inference_raw_image_path = os.path.join(
            save_path, "vlm_inference_raw_image"
        )
        gaze_point_target_mask_image_path = os.path.join(
            save_path, "gaze_point_target_mask_image"
        )
        os.makedirs(gaze_point_target_mask_image_path, exist_ok=True)
        with open(
            os.path.join(save_path, "gaze_point_target_bbox.txt"), "r", encoding="utf-8"
        ) as f:
            cnt = 0
            while True:
                line = f.readline()
                if line == "":
                    break
                bbox_xyxy = line.split()
                if len(bbox_xyxy) == 0:
                    image_viz = cv2.imread(
                        os.path.join(vlm_inference_raw_image_path, f"{cnt}.jpg")
                    )
                    if self.viz_flag:
                        cv2.imshow("image_viz", image_viz)
                        cv2.waitKey(1000)
                    cv2.imwrite(
                        os.path.join(gaze_point_target_mask_image_path, f"{cnt}.jpg"),
                        image_viz,
                    )
                    cnt += 1
                    continue
                image_raw = cv2.imread(
                    os.path.join(vlm_inference_raw_image_path, f"{cnt}.jpg")
                )
                image_cv2_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                x1 = int(bbox_xyxy[0])
                y1 = int(bbox_xyxy[1])
                x2 = int(bbox_xyxy[2])
                y2 = int(bbox_xyxy[3])
                bbox = torch.tensor([[x1, y1, x2, y2]])
                self.sam_model.set_image(image_cv2_rgb)
                transformed_boxes = self.sam_model.transform.apply_boxes_torch(
                    bbox, image_cv2_rgb.shape[:2]
                ).to(self.alg_args.device)
                masks, _, _ = self.sam_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                mask = masks.cpu().squeeze(1).numpy()[0]
                image_viz = self._draw_masks(mask, image_raw)
                # image_viz = cv2.rectangle(
                #     image_viz,
                #     (x1, y1),
                #     (x2, y2),
                #     (0, 0, 255),
                #     thickness=2,
                # )
                if self.viz_flag:
                    cv2.imshow("image_viz", image_viz)
                    cv2.waitKey(1000)
                cv2.imwrite(
                    os.path.join(gaze_point_target_mask_image_path, f"{cnt}.jpg"),
                    image_viz,
                )

                cnt += 1

    def _draw_masks(self, mask, image_viz):
        mask_map = np.zeros(image_viz.shape, dtype=np.uint8)
        mask_color = (0, 0, 255)
        mask_map[mask == True] = mask_color  # 这里mask is True不行，只能==True
        image_viz = cv2.addWeighted(
            image_viz, 1, mask_map, self.alg_args.mask_opacity, gamma=0
        )
        return image_viz


def main():
    rospy.init_node("add_mask")
    AddMask()
    cv2.destroyAllWindows()
    print("Finished!")


if __name__ == "__main__":
    main()
