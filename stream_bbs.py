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

def detect_objects(frame):
    original_height, original_width = frame.shape[:2]
    input_height, input_width = interpreter.get_input_details()[0]['shape'][1:3]

    # Resize while maintaining aspect ratio and adding padding if needed
    scale = min(input_width / original_width, input_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Calculate padding
    pad_x = (input_width - new_width) // 2
    pad_y = (input_height - new_height) // 2

    # Pad the resized image
    padded_frame = cv2.copyMakeBorder(resized_frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Set the input tensor
    common.set_input(interpreter, padded_frame)

    # Run inference
    interpreter.invoke()

    # Get detected objects
    boxes = detect.get_objects(interpreter, score_threshold=0.5)

    for obj in boxes:
        ymin, xmin, ymax, xmax = obj.bbox

        # Undo padding and scaling to get coordinates in original image
        xmin = max(0, int((xmin - pad_x) / scale))
        xmax = min(original_width, int((xmax - pad_x) / scale))
        ymin = max(0, int((ymin - pad_y) / scale))
        ymax = min(original_height, int((ymax - pad_y) / scale))

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Draw label
        label = COCO_LABELS[obj.id] if obj.id < len(COCO_LABELS) else 'Unknown'
        label = f'{label}: {obj.score:.2f}'
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

class VLCPlayer:
    def __init__(self, url):
        self.url = url
        self.instance = vlc.Instance("--no-audio", "--no-xlib", "--video-title-show", "--no-video-title")
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new(self.url)
        self.player.set_media(self.media)
        self.width = 640
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

    cv2.namedWindow("Video Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame = player.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Apply object detection
        detected_frame = detect_objects(frame_rgb)

        cv2.imshow("Video Stream", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    player.player.stop()
