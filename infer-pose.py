from models import TRTModule
import argparse
import time
from pathlib import Path

import cv2
import torch

from config import COLORS, KPS_COLORS, LIMB_COLORS, SKELETON
from models.torch_utils import pose_postprocess
from models.utils import blob, letterbox


def main() -> None:
    device = torch.device('cuda:0')
    Engine = TRTModule('yolov8s-pose.engine', device)
    H, W = Engine.inp_info[0].shape[-2:]

    cap = cv2.VideoCapture(0)  # 使用默认摄像头，如果有多个摄像头，可以尝试不同的索引值

    frame_count = 0
    start_time = time.time()

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break

        bgr = frame.copy()
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, kpts = pose_postprocess(data, 0.25, 0.65)
        if bboxes.numel() == 0:
            # if no bounding box
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, kpt) in zip(bboxes, scores, kpts):
            bbox = bbox.round().int().tolist()
            color = COLORS['person']
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'person:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
            for i in range(19):
                if i < 17:
                    px, py, ps = kpt[i]
                    if ps > 0.5:
                        kcolor = KPS_COLORS[i]
                        px = round(float(px - dw) / ratio)
                        py = round(float(py - dh) / ratio)
                        cv2.circle(draw, (px, py), 5, kcolor, -1)
                xi, yi = SKELETON[i]
                pos1_s = kpt[xi - 1][2]
                pos2_s = kpt[yi - 1][2]
                if pos1_s > 0.5 and pos2_s > 0.5:
                    limb_color = LIMB_COLORS[i]
                    pos1_x = round(float(kpt[xi - 1][0] - dw) / ratio)
                    pos1_y = round(float(kpt[xi - 1][1] - dh) / ratio)
                    pos2_x = round(float(kpt[yi - 1][0] - dw) / ratio)
                    pos2_y = round(float(kpt[yi - 1][1] - dh) / ratio)
                    cv2.line(draw, (pos1_x, pos1_y),
                             (pos2_x, pos2_y), limb_color, 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(draw, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('result', draw)
        if cv2.waitKey(1) == ord('q'):  # 按下 q 键退出循环
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
