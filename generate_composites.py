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

def create_masks_with_tflite(path_to_video_file, output_frame_memmaps, output_frame_mask_memmaps, model_path):
    """
    Process each frame of the video using the TFLite model for segmentation,
    and save the masked frames into memory-mapped files.
    """
    # Load the TFLite model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Open the video file
    cap = cv2.VideoCapture(path_to_video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create memory-mapped files to save the masked frames
    memmap_frames = np.memmap(output_frame_memmaps, dtype='uint8', mode='w+', shape=(num_frames, frame_height, frame_width, 3))
    memmap_masks = np.memmap(output_frame_mask_memmaps, dtype='uint8', mode='w+', shape=(num_frames, frame_height, frame_width))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask to the current frame (only for "person")
        masked_frame = mask_frame(frame, interpreter, selected_label="person")

        # Save the masked frame and the mask to the memory-mapped files
        memmap_frames[frame_idx] = masked_frame
        memmap_masks[frame_idx] = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)
        frame_idx += 1

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

    TMP_VIDEO_PATH = "_new_video.mp4"
    NEW_MASK_MEMMAP_PATH = "_new_video.dat"
    COMPOSITES_DIR = "composites"
    MODEL_PATH = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'

    # Clear composites from the composites directory because they might not match
    # the position of the camera now.
    reset_directory(COMPOSITES_DIR)

    # Record a new video.
    print(f"--- Starting up... Recording some new video: {TMP_VIDEO_PATH}")
    recorder = RTSPVideoRecorder(rtsp_url=RTSP_URL,
                                 output_file=TMP_VIDEO_PATH,
                                 duration=DURATION,
                                 fps=FPS)
    recorder.record()

    # Convert the video to a masked memmap file using TFLite model.
    print(f"--- Converting the video to a masked memmap: {NEW_MASK_MEMMAP_PATH}")
    create_masks_with_tflite(path_to_video_file=TMP_VIDEO_PATH,
                             output_frame_memmaps=NEW_MASK_MEMMAP_PATH,
                             output_frame_mask_memmaps="_new_video_mask.dat",
                             model_path=MODEL_PATH)
    
    # Copy mask to the composites directory.
    composite_memmap_copy_path = os.path.join(COMPOSITES_DIR, NEW_MASK_MEMMAP_PATH)
    print(f"--- Copying {NEW_MASK_MEMMAP_PATH} to {composite_memmap_copy_path}")
    shutil.copy(NEW_MASK_MEMMAP_PATH, composite_memmap_copy_path)


    while True:
        # Get the time for file naming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        print("1) Recording video")
        # Record 5 seconds of video
        recorder = RTSPVideoRecorder(rtsp_url=RTSP_URL,
                                     output_file=TMP_VIDEO_PATH,
                                     duration=DURATION,
                                     fps=FPS)
        recorder.record()

        # Create masks for the video using TFLite model
        print(f"2) Creating mask: {NEW_MASK_MEMMAP_PATH}")
        create_masks_with_tflite(path_to_video_file=TMP_VIDEO_PATH,
                                 output_frame_memmaps=NEW_MASK_MEMMAP_PATH,
                                 output_frame_mask_memmaps="_new_video_mask.dat",
                                 model_path=MODEL_PATH)

        # Overlay with the previously created composite video
        most_recent_composite = get_most_recent_file(COMPOSITES_DIR)
        output_memmap_path = os.path.join(COMPOSITES_DIR, current_time) + ".dat"

        print(f"3) Creating {output_memmap_path} from {most_recent_composite} and {NEW_MASK_MEMMAP_PATH}")
        overlay_videos(background_video_memmap=most_recent_composite,
                       foreground_video_memmap=NEW_MASK_MEMMAP_PATH,
                       output_video_memmap=output_memmap_path,
                       height=HEIGHT,
                       width=WIDTH)
        
        print("--------------------------------------------------------------")
