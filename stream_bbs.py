import os
import vlc
import numpy as np
import cv2
import ctypes
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

# Initialize the Coral Edge TPU with the SSD MobileNet model
model_path = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# COCO labels for the object detection categories
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

def resize_with_aspect_ratio(image, target_size):
    """
    Resize the image while keeping the aspect ratio consistent. Add padding if necessary.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate the scaling factor
    scale = min(target_w / w, target_h / h)

    # Compute the new dimensions and padding
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    delta_w = target_w - new_w
    delta_h = target_h - new_h

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add padding to get the target size
    color = [0, 0, 0]
    resized_with_padding = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return resized_with_padding, scale, (left, top)

def detect_objects(frame):
    """
    Perform object detection on the frame and map the bounding boxes back to the original frame.
    """
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    original_height, original_width = frame.shape[:2]

    # Resize with aspect ratio preserved
    resized_frame, scale, (pad_x, pad_y) = resize_with_aspect_ratio(frame, (input_width, input_height))

    # Set the input tensor
    common.set_input(interpreter, resized_frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and postprocess to obtain detections
    boxes = detect.get_objects(interpreter, score_threshold=0.5)

    for obj in boxes:
        ymin, xmin, ymax, xmax = obj.bbox

        # Remove padding and scale back to original size
        xmin = int((xmin - pad_x) / scale)
        xmax = int((xmax - pad_x) / scale)
        ymin = int((ymin - pad_y) / scale)
        ymax = int((ymax - pad_y) / scale)

        # Ensure bounding box stays within image bounds
        xmin = max(0, min(xmin, original_width))
        xmax = max(0, min(xmax, original_width))
        ymin = max(0, min(ymin, original_height))
        ymax = max(0, min(ymax, original_height))

        # Draw bounding box (red color, thickness 1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

        # Get the label name from COCO_LABELS
        label = COCO_LABELS[obj.id] if obj.id < len(COCO_LABELS) else 'Unknown'
        label = f'{label}: {obj.score:.2f}'

        # Draw the label above the bounding box
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

class VLCPlayer:
    def __init__(self, url):
        self.url = url
        self.instance = vlc.Instance("--no-audio", "--no-xlib", "--video-title-show", "--no-video-title")
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new(self.url)
        self.player.set_media(self.media)
        self.width = 640  # Set according to your stream resolution
        self.height = 480
        self.frame_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.frame_pointer = self.frame_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.setup_vlc()

    def setup_vlc(self):
        self.lock_cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(self.lock_cb)
        self.unlock_cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(self.unlock_cb)
        self.display_cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(self.display_cb)
        
        self.player.video_set_callbacks(self.lock_cb, self.unlock_cb, self.display_cb, None)
        self.player.video_set_format("RV32", self.width, self.height, self.width * 4)

    def lock_cb(self, opaque, planes):
        planes[0] = ctypes.cast(self.frame_pointer, ctypes.c_void_p)

    def unlock_cb(self, opaque, picture, planes):
        pass

    def display_cb(self, opaque, picture):
        pass

    def start(self):
        self.player.play()

    def get_frame(self):
        return np.copy(self.frame_data)


if __name__ == "__main__":
    os.environ['DISPLAY'] = ':0'

    url = "https://61e0c5d388c2e.streamlock.net/live/MLK_E_Cherry_NS.stream/chunklist_w1373546751.m3u8"
    player = VLCPlayer(url)
    player.start()

    # Create a named window and set it to full screen
    cv2.namedWindow("Video Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame = player.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Apply object detection
        detected_frame = detect_objects(frame_rgb)

        # Display the frame in full screen
        cv2.imshow("Video Stream", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    player.player.stop()
