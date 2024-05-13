#!/usr/bin/env python3
import asyncio
import base64
import gzip
import logging
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sport_pose_estimation import PoseEstimator

app = FastAPI()

# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='pose_landmarker.log')

# 存储连接的WebSocket客户端
connected_clients = set()


async def send_frame(websocket: WebSocket, estimator: PoseEstimator):
    while estimator.cap.isOpened():
        # 从视频中读取一帧
        success, frame = estimator.cap.read()
        if success:
            jpeg = estimator.process_frame(frame)
            if jpeg is not None:
                await websocket.send_text(jpeg)
                # await websocket.send_bytes(jpeg)
                await asyncio.sleep(0.01)


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    # 获取查询参数中的 sport 值
    sport = websocket.query_params.get('sport', 'squat')
    # 初始化 PoseEstimator
    estimator = PoseEstimator(sport=sport)

    try:
        await send_frame(websocket, estimator)
    except WebSocketDisconnect:
        # 处理客户端断开连接的情况
        handle_disconnect(websocket, estimator)
    except Exception as e:
        # 处理其他异常
        logging.error(f"Error occurred: {e}")
        handle_disconnect(websocket, estimator)


def handle_disconnect(websocket: WebSocket, estimator: PoseEstimator):
    if estimator.cap.isOpened():
        # 释放视频捕捉对象并关闭显示窗口
        estimator.cap.release()
        if estimator.save_dir is not None:
            estimator.output.release()
        cv2.destroyAllWindows()

    # 断开连接时从已连接的客户端列表中删除客户端
    connected_clients.remove(websocket)


# 运行 FastAPI 应用
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
