import time
import cv2
import numpy as np
import yaml
from image_utils import get_most_recent_file
import os
import threading
import queue

def get_num_frames(memmap_file, height, width, channels=3):
    file_size = np.memmap(memmap_file, dtype='uint8', mode='r').shape[0]
    frame_size = height * width * channels

    return file_size // frame_size

def load_overlay_file(overlay_memmap_file, height, width, channels=3):
    num_overlay_frames = get_num_frames(overlay_memmap_file, height, width, channels)
    overlay_frames = np.memmap(overlay_memmap_file,
                               dtype='uint8',
                               mode='r',
                               shape=(num_overlay_frames, height, width, channels))
    
    return overlay_frames, num_overlay_frames

def capture_frames(cap, frame_queue, width, height, fps):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(1 / fps)  # Sync with expected FPS

def display_frames(frame_queue, overlay_frames, num_overlay_frames):
    overlay_index = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            overlay_frame = overlay_frames[overlay_index]

            overlay_mask = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2GRAY)
            _, overlay_mask = cv2.threshold(overlay_mask, 1, 255, cv2.THRESH_BINARY)
            rtsp_mask = cv2.bitwise_not(overlay_mask)

            rtsp_frame_masked = cv2.bitwise_and(frame, frame, mask=rtsp_mask)
            overlay_frame_masked = cv2.bitwise_and(overlay_frame, overlay_frame, mask=overlay_mask)
            combined_frame = cv2.add(rtsp_frame_masked, overlay_frame_masked)

            cv2.imshow("Combined Frame", combined_frame)

            overlay_index = (overlay_index + 1) % num_overlay_frames

            if cv2.waitKey(1) == 27:
                break

        time.sleep(0.01)

def stream_and_overlay(rtsp_url_or_camera_index, height, width, composites_dir, channels=3, fps=15):
    cap = cv2.VideoCapture(rtsp_url_or_camera_index)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    overlay_memmap_file = get_most_recent_file(composites_dir)
    print(f"Overlaying video stream with: {overlay_memmap_file}")
    overlay_frames, num_overlay_frames = load_overlay_file(overlay_memmap_file,
                                                           height,
                                                           width,
                                                           channels)

    frame_queue = queue.Queue(maxsize=10)

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, width, height, fps))
    capture_thread.start()

    display_frames(frame_queue, overlay_frames, num_overlay_frames)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['DISPLAY'] = ':0'

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RTSP_URL_OR_CAMERA_INDEX = config["rtsp_url"]
    HEIGHT = config["height"]
    WIDTH = config["width"]
    DURATION = config["duration"]
    FPS = config["fps"]

    stream_and_overlay(rtsp_url_or_camera_index=RTSP_URL_OR_CAMERA_INDEX,
                       height=HEIGHT,
                       width=WIDTH,
                       composites_dir="composites",
                       fps=FPS)
