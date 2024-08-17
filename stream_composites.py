import time
import cv2
import numpy as np
import yaml
from image_utils import get_most_recent_file
import os



def get_num_frames(memmap_file, height, width, channels=3):
    """
    Calculate the number of frames based on the size of the memmap file.
    """
    file_size = np.memmap(memmap_file, dtype='uint8', mode='r').shape[0]
    frame_size = height * width * channels

    return file_size // frame_size

def load_overlay_file(overlay_memmap_file, height, width, channels=3):
    """
    Load the memmap file with overlay frames.
    """
    num_overlay_frames = get_num_frames(overlay_memmap_file, height, width, channels)
    overlay_frames = np.memmap(overlay_memmap_file,
                               dtype='uint8',
                               mode='r',
                               shape=(num_overlay_frames, height, width, channels))
    
    return overlay_frames, num_overlay_frames

def stream_and_overlay(rtsp_url_or_camera_index, height, width, composites_dir, channels=3):
    """
    Open an RTSP stream or local camera and overlay the frames with frames from the `composites_dir`.
    The composites should be created from frames from the same camera in the same position (the camera must be stationary).
    """
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
    
    overlay_index = 0

    while True:
        
        os.system("xset dpms force on")
        # Read a frame from the video capture
        ret, rtsp_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            break

        # Resize the frame to match the expected height and width
        rtsp_frame = cv2.resize(rtsp_frame, (width, height))

        # Get the current overlay frame
        overlay_frame = overlay_frames[overlay_index]

        # Ensure both frames have the same number of channels
        if rtsp_frame.shape[2] != overlay_frame.shape[2]:
            if rtsp_frame.shape[2] == 3 and overlay_frame.shape[2] == 1:
                overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_GRAY2BGR)
            elif rtsp_frame.shape[2] == 1 and overlay_frame.shape[2] == 3:
                rtsp_frame = cv2.cvtColor(rtsp_frame, cv2.COLOR_BGR2GRAY)

        # Create a binary mask from the overlay by checking where there are non-black pixels
        overlay_mask = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2GRAY)
        _, overlay_mask = cv2.threshold(overlay_mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the mask to create a mask for the RTSP frame
        rtsp_mask = cv2.bitwise_not(overlay_mask)

        # Ensure the mask dimensions match the RTSP frame
        if rtsp_frame.shape[:2] != rtsp_mask.shape:
            rtsp_mask = cv2.resize(rtsp_mask, (rtsp_frame.shape[1], rtsp_frame.shape[0]))

        # Apply the masks
        rtsp_frame_masked = cv2.bitwise_and(rtsp_frame, rtsp_frame, mask=rtsp_mask)
        overlay_frame_masked = cv2.bitwise_and(overlay_frame, overlay_frame, mask=overlay_mask)

        # Combine the frames
        combined_frame = cv2.add(rtsp_frame_masked, overlay_frame_masked)

        # Display the combined frame
        cv2.imshow("Combined Frame", combined_frame)

        # Increment the overlay index
        overlay_index += 1
        
        # Check if we've exhausted the current overlay frames
        if overlay_index >= num_overlay_frames:
            # Reset the index
            overlay_index = 0
            
            # Check for a new most recent file
            new_overlay_memmap_file = get_most_recent_file(composites_dir)
            if new_overlay_memmap_file != overlay_memmap_file:
                print(f"Switching to new overlay file: {new_overlay_memmap_file}")
                overlay_memmap_file = new_overlay_memmap_file
                overlay_frames, num_overlay_frames = load_overlay_file(overlay_memmap_file,
                                                                       height,
                                                                       width,
                                                                       channels)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            break

        # Short delay to allow for checking the new most recent file periodically
        time.sleep(0.01)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    os.environ['DISPLAY'] = ':0'

    # Read the config file.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RTSP_URL_OR_CAMERA_INDEX = config["rtsp_url"]  # Can be an RTSP URL or an integer camera index
    HEIGHT = config["height"]
    WIDTH = config["width"]
    DURATION = config["duration"]
    FPS = config["fps"]

    stream_and_overlay(rtsp_url_or_camera_index=RTSP_URL_OR_CAMERA_INDEX,
                       height=HEIGHT,
                       width=WIDTH,
                       composites_dir="composites")
