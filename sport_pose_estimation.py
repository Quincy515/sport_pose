import base64
import os
import cv2
import numpy as np
import math
import datetime
import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy

# 定义运行及运动参数
sport_list = {
    'sit-up': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]]
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]]
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [15, 13], [13, 11]]
    },
    'jump': {
        'left_points_idx': [11, 13, 15],  # 左侧关键点索引
        'right_points_idx': [12, 14, 16],  # 右侧关键点索引
        'maintaining': 20,  # 维持姿势的度量
        'relaxing': 140,  # 放松时的度量
        'concerned_key_points_idx': [11, 12, 13, 14, 15],  # 关注的关键点索引
        # 关注的骨架索引
        'concerned_skeletons_idx': [[16, 14], [14, 12], [15, 13], [13, 11]]
    },
    # 'diff': {
    #     'left_points_idx': [11],  # 左侧关键点索引
    #     'right_points_idx': [12],  # 右侧关键点索引
    #     'maintaining': 20,  # 维持姿势的度量
    #     'relaxing': 140,  # 放松时的度量
    #     'concerned_key_points_idx': [11, 12, 13, 14, 15],
    #     'concerned_skeletons_idx': [[16, 14], [14, 12], [15, 13], [13, 11]]
    # },
    'diff': {
        'left_points_idx': [11, 13, 15],  # 左侧关键点索引
        'right_points_idx': [12, 14, 16],  # 右侧关键点索引
        'maintaining': 20,  # 维持姿势的度量
        'relaxing': 140,  # 放松时的度量
        'concerned_key_points_idx': [11, 12, 13, 14, 15],  # 关注的关键点索引
        # 关注的骨架索引
        'concerned_skeletons_idx': [[16, 14], [14, 12], [15, 13], [13, 11]]
    },
}


def calculate_angle(key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        # 计算两条直线的斜率
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # 将斜率转换为角度
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # 计算角度差
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]]
                   for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]]
                    for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
    return angle


def calculate_angle_filter(sports_name, key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        # 计算两条直线的斜率
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # 将斜率转换为角度
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # 计算角度差
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]]
                   for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]]
                    for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    if (sports_name == 'diff'):
        angle = max(angle_left, angle_right)
    else:
        angle = (angle_left + angle_right) / 2
    return angle


def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
    class _Annotator(Annotator):

        def kpts(self, kpts, shape=(640, 640), radius=5, line_thickness=2, kpt_line=True):
            """Plot keypoints on the image.

            Args:
                kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
                shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
                radius (int, optional): Radius of the drawn keypoints. Default is 5.
                kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                           for human pose. Default is True.
                line_thickness (int, optional): thickness of the kpt_line. Default is 2.

            Note: `kpt_line=True` currently only supports human pose plotting.
            """
            if self.pil:
                # Convert to numpy first
                self.im = np.asarray(self.im).copy()
            nkpt, ndim = kpts.shape
            is_pose = nkpt == 17 and ndim == 3
            kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
            colors = Colors()
            for i, k in enumerate(kpts):
                if show_points is not None:
                    if i not in show_points:
                        continue
                color_k = [int(x) for x in self.kpt_color[i]
                           ] if is_pose else colors(i)
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)),
                               int(radius * plot_size_redio), color_k, -1, lineType=cv2.LINE_AA)

            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    if show_skeleton is not None:
                        if sk not in show_skeleton:
                            continue
                    pos1 = (int(kpts[(sk[0] - 1), 0]),
                            int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]),
                            int(kpts[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             thickness=int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))
    if pose_result.keypoints is not None:
        for k in reversed(pose_result.keypoints.data):
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)
    return annotator.result()


def put_text(frame, exercise, count, fps, redio):
    # 在图像帧上绘制一个矩形作为文本背景
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)
                ), (int(300 * redio), int(163 * redio)),
        (55, 104, 0), -1
    )

    # 如果运动在运动列表中，将运动名称添加到帧上
    if exercise in sport_list.keys():
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(
                30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    # 如果没有检测到对象，将"No Object"添加到帧上
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio),
                                  int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    # 将计数添加到帧上
    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio),
                                   int(100 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
    # 将FPS添加到帧上
    cv2.putText(
        frame, f'FPS: {fps}', (int(30 * redio),
                               int(150 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )


class PoseEstimator:
    def __init__(self, model='yolov8s-pose.pt', sport='squat', input="0", save_dir=None, show=False):
        self.cap = None
        self.save_dir = None
        self.output = None

        # 设置GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'使用设备: {self.device}')

        # 加载YOLOv8模型
        self.model = YOLO(model).to(self.device)

        # 打开视频文件或摄像头
        if input.isnumeric():
            self.cap = cv2.VideoCapture(int(input))
            # 选择清晰度, 要确保所设置的分辨率是摄像头支持的
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # 设置帧宽度为 1080
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)  # 设置帧高度为 1920
            # 检查是否成功设置清晰度
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"当前摄像头分辨率：{width}x{height}")
        else:
            self.cap = cv2.VideoCapture(input)

        # 保存结果视频
        if save_dir is not None:
            self.save_dir = os.path.join(
                save_dir, sport,
                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.output = cv2.VideoWriter(os.path.join(
                self.save_dir, 'result.mp4'), fourcc, fps, size)
        else:
            self.output = None

        self.sport = sport
        self.show = show

        # 设置记录运动状态的变量
        self.reaching = False
        self.reaching_last = False
        self.state_keep = False
        self.counter = 0
        self.flip_list = []

    def process_frame(self, frame):
        # 设置不同分辨率输入的绘图大小比例
        plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

        # 在帧上运行YOLOv8推断
        results = self.model(frame)

        # 防止特殊情况导致的错误
        if results[0].keypoints.shape[1] == 0:
            put_text(frame, '无目标', self.counter,
                     round(1000 / results[0].speed['inference'], 2), plot_size_redio)
            scale = 640 / max(frame.shape[0], frame.shape[1])
            show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            if self.show:
                cv2.imshow("YOLOv8推断", show_frame)
            else:
                _, imencode_image = cv2.imencode(".jpg", show_frame)
                return imencode_image.tobytes()

            if self.save_dir is not None:
                self.output.write(frame)
            return

        # 获取运动参数
        left_points_idx = sport_list[self.sport]['left_points_idx']
        right_points_idx = sport_list[self.sport]['right_points_idx']

        # 如果是 diff 运动，通过波峰波谷计数
        '''
        if (self.sport != 'diff'):
            # 计算角度
            angle = calculate_angle(
                results[0].keypoints, left_points_idx, right_points_idx)

            # 确定是否完成一次
            if angle < sport_list[self.sport]['maintaining']:
                self.reaching = True
            if angle > sport_list[self.sport]['relaxing']:
                self.reaching = False

            if self.reaching != self.reaching_last:
                self.reaching_last = self.reaching
                if self.reaching:
                    self.state_keep = True
                if not self.reaching and self.state_keep:
                    self.counter += 1
                    self.state_keep = False
        elif (self.sport == 'diff'):
            # 保存计算左右髋关节的平均高度值
            left_points = [results[0].keypoints.data[0][i][1]
                           for i in left_points_idx]
            right_points = [results[0].keypoints.data[0][i][1]
                            for i in right_points_idx]
            flip = (sum(left_points) + sum(right_points)) / 2
            # 把平均高度值保存到数组
            self.flip_list.append(flip)
            # 如果超过2个点就开始比较
            if len(self.flip_list) >= 2:
                prev_flip = self.flip_list[len(self.flip_list) - 2]
                # 开始进行判断计数
                if (flip <= prev_flip or self.state_keep != False) and (flip >= prev_flip or self.state_keep != True):
                    count = count + 1
                    self.state_keep = not self.state_keep
            self.counter = int(count/2)
        '''
        angle = calculate_angle_filter(self.sport,
                                       results[0].keypoints, left_points_idx, right_points_idx)

        # 确定是否完成一次
        if angle < sport_list[self.sport]['maintaining']:
            self.reaching = True
        if angle > sport_list[self.sport]['relaxing']:
            self.reaching = False

        if self.reaching != self.reaching_last:
            self.reaching_last = self.reaching
            if self.reaching:
                self.state_keep = True
            if not self.reaching and self.state_keep:
                self.counter += 1
                self.state_keep = False
        # 在帧上可视化结果
        annotated_frame = plot(
            results[0], plot_size_redio,
            # sport_list[sport]['concerned_key_points_idx'],
            # sport_list[sport]['concerned_skeletons_idx']
        )
        # annotated_frame = results[0].plot(boxes=False)

        # 添加相关信息到帧
        put_text(
            annotated_frame, self.sport, self.counter, round(1000 / results[0].speed['inference'], 2), plot_size_redio)

        if self.save_dir is not None:
            self.output.write(annotated_frame)

        # 显示注释帧
        if self.show:
            scale = 640 / \
                max(annotated_frame.shape[0], annotated_frame.shape[1])
            show_frame = cv2.resize(
                annotated_frame, (0, 0), fx=scale, fy=scale)
            cv2.imshow("YOLOv8推断", show_frame)
        else:
            # 调整显示图片尺寸
            scale = 640 / \
                max(annotated_frame.shape[0], annotated_frame.shape[1])
            show_frame = cv2.resize(
                annotated_frame, (1080, 1920), fx=scale, fy=scale)
            # 如果 show 为 False,则通过 WebSocket 发送处理后的帧
            # 转换为 jpeg 格式
            _, imencode_image = cv2.imencode(".jpg", show_frame)
            # base64_image = base64.b64encode(
            #     imencode_image).decode("utf-8")
            return imencode_image.tobytes()

    def process(self):
        # 循环视频帧
        while self.cap.isOpened():
            # 从视频读取一帧
            success, frame = self.cap.read()

            if success:
                self.process_frame(frame)
                # 如果按下'q'，则中断循环
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # 如果到达视频的结尾，中断循环
                break

        # 释放视频捕获对象并关闭显示窗口
        self.cap.release()
        if self.save_dir is not None:
            self.output.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    estimator = PoseEstimator(show=True)
    estimator.process()
