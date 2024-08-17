import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
import os
import shutil
import yaml
from datetime import datetime
from image_utils import RTSPVideoRecorder, get_most_recent_file
from overlay_memmaps import overlay_videos

# List of category labels for the PASCAL VOC 2012 dataset
LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

def create_pascal_label_colormap():
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    """
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap

def mask_frame(frame, interpreter, selected_label):
    """
    Apply semantic segmentation on the input frame, filter by selected label,
    and remove the background.
    """
    original_height, original_width = frame.shape[:2]
    width, height = common.input_size(interpreter)

    # Resize the input frame to match model input size
    img = Image.fromarray(frame)
    resized_img = img.resize((width, height), Image.LANCZOS)
    common.set_input(interpreter, resized_img)

    # Run inference
    interpreter.invoke()

    # Get the segmentation result
    result = segment.get_output(interpreter)
    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

    # Filter the segmentation result to only include the selected label
    mask = np.zeros_like(result)
    label_index = LABELS.index(selected_label)
    mask[result == label_index] = 255

    # Resize the mask to match the original frame dimensions
    mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_resized)

    return masked_frame

def create_masks_with_tflite(recording_duration, masked_memmaps_path, model_path, height, width, rtsp_url, fps):
    """
    Capture frames from the RTSP stream for a specified duration, apply TFLite model for segmentation,
    and save the masked frames into a memory-mapped file.
    """
    # Initialize the interpreter for the TFLite model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Calculate the number of frames to capture based on duration and FPS
    num_frames = int(recording_duration * fps)

    # Create a memory-mapped file for the masked frames
    memmap_frames = np.memmap(masked_memmaps_path, dtype='uint8', mode='w+', shape=(num_frames, height, width, 3))

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from RTSP stream.")
            break

        # Resize the frame to match the target dimensions (height x width)
        frame_resized = cv2.resize(frame, (width, height))

        # Apply the TFLite model to mask the frame
        masked_frame = mask_frame(frame_resized, interpreter, selected_label="person")
        memmap_frames[frame_idx] = masked_frame
        frame_idx += 1

    # Flush the output memory-mapped file to disk
    memmap_frames.flush()

    # Release the RTSP stream
    cap.release()


def reset_directory(directory_path):
    """
    Reset the directory by removing it and all its contents, then recreating it.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    
    os.makedirs(directory_path)

if __name__ == "__main__":
    # Read the config file.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RTSP_URL = config["rtsp_url"]
    HEIGHT = config["height"]
    WIDTH = config["width"]
    DURATION = config["duration"]
    FPS = config["fps"]

    COMPOSITES_DIR = "composites"
    MODEL_PATH = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'

    # Clear composites from the composites directory because they might not match the position of the camera now.
    reset_directory(COMPOSITES_DIR)

    # Record a new video as a memmap
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_memmap_path = os.path.join(COMPOSITES_DIR, current_time) + ".dat"
    print(f"Saving masked video frames as a memmap: {output_memmap_path}")
    create_masks_with_tflite(recording_duration=DURATION,
                             masked_memmaps_path=output_memmap_path,
                             model_path=MODEL_PATH,
                             height=HEIGHT,
                             width=WIDTH,
                             rtsp_url=RTSP_URL,
                             fps=FPS)
    

    # # Copy mask to the composites directory.
    # composite_memmap_copy_path = os.path.join(COMPOSITES_DIR, NEW_MASK_MEMMAP_PATH)
    # print(f"--- Copying {NEW_MASK_MEMMAP_PATH} to {composite_memmap_copy_path}")
    # shutil.copy(NEW_MASK_MEMMAP_PATH, composite_memmap_copy_path)










    # while True:
    #     # Get the time for file naming
    #     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    #     print("1) Recording video as memmap")

    #     print(f"2) Creating mask: {NEW_MASK_MEMMAP_PATH}")
    #     create_masks_with_tflite(path_to_input_memmaps,
    #                              output_frame_memmaps=NEW_MASK_MEMMAP_PATH,
    #                              output_frame_mask_memmaps="_new_video_mask.dat",
    #                              model_path=MODEL_PATH)

    #     # Overlay with the previously created composite video
    #     most_recent_composite = get_most_recent_file(COMPOSITES_DIR)
    #     output_memmap_path = os.path.join(COMPOSITES_DIR, current_time) + ".dat"

    #     print(f"3) Creating {output_memmap_path} from {most_recent_composite} and {NEW_MASK_MEMMAP_PATH}")
    #     overlay_videos(background_video_memmap=most_recent_composite,
    #                    foreground_video_memmap=NEW_MASK_MEMMAP_PATH,
    #                    output_video_memmap=output_memmap_path,
    #                    height=HEIGHT,
    #                    width=WIDTH)
        
    #     print("--------------------------------------------------------------")
