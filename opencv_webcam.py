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
    """
    Sends frames from a video stream over a WebSocket connection.

    Args:
        websocket (WebSocket): The WebSocket connection to send the frames to.

    Returns:
        None
    """
    while estimator.cap.isOpened():
        # Read a frame from the video
        success, frame = estimator.cap.read()

        if success:
            jpeg = estimator.process_frame(frame)
            if jpeg is not None:
                await websocket.send_bytes(jpeg)
                await asyncio.sleep(0.01)


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming video frames.

    Parameters:
    - websocket: WebSocket object representing the connection with the client.

    Returns:
    - None

    Raises:
    - WebSocketDisconnect: If the client disconnects unexpectedly.
    """
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
        # Remove the client from the connected clients list when disconnected
        connected_clients.remove(websocket)

# 关闭 websocket 连接,释放资源


@app.websocket("/ws/close")
async def close_websocket(websocket: WebSocket):
    if estimator.cap.isOpened():
        # Release the video capture object and close the display window
        estimator.cap.release()
        if estimator.save_dir is not None:
            estimator.output.release()
        cv2.destroyAllWindows()
    await websocket.close()

# 运行 FastAPI 应用
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
