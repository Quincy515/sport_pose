#!/usr/bin/env python3
import asyncio
import base64
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
estimator = PoseEstimator()


async def send_frame(websocket: WebSocket):
    while estimator.cap.isOpened():
        # Read a frame from the video
        success, frame = estimator.cap.read()

        if success:
            jpeg = estimator.process_frame(frame)
            await websocket.send_text(jpeg)
            await asyncio.sleep(0.01)


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        await send_frame(websocket)
    except WebSocketDisconnect:
        if estimator.cap.isOpened():
            # Release the video capture object and close the display window
            estimator.cap.release()
            if estimator.save_dir is not None:
                estimator.output.release()
            cv2.destroyAllWindows()
        # 客户端断开连接时，从存储中移除
        connected_clients.remove(websocket)


# 运行 FastAPI 应用
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
