import os
import vlc
import numpy as np
import cv2
import ctypes
from pycoral.adapters import common, detect, segment
from pycoral.utils.edgetpu import make_interpreter



# Download the model
model_path = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'

# Initialize the Coral Edge TPU with the SSD MobileNet model
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

def detect_objects(frame):
    # Resize the input frame to match the model's expected input size
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Set the input tensor
    common.set_input(interpreter, resized_frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and postprocess to obtain detections
    boxes = detect.get_objects(interpreter, score_threshold=0.5)

    # Draw bounding boxes on the original frame
    for obj in boxes:
        ymin, xmin, ymax, xmax = obj.bbox
        ymin = int(ymin * frame.shape[0] / input_height)
        xmin = int(xmin * frame.shape[1] / input_width)
        ymax = int(ymax * frame.shape[0] / input_height)
        xmax = int(xmax * frame.shape[1] / input_width)
        
        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f'{obj.id}: {obj.score:.2f}'
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame



def mask_frame(frame):
    # Resize the input frame to match the model's expected input size
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Set the input tensor
    common.set_input(interpreter, resized_frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    result = segment.get_output(interpreter)

    # Resize the result to match the original frame size
    mask = cv2.resize(result, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create an overlay for the mask using the fixed color map
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # Skip the background
        overlay[mask == class_id] = COLOR_MAP[class_id]

    # Combine the original frame with the mask overlay using transparency
    alpha = 0.4
    masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return masked_frame


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

    while True:
        frame = player.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        masked_frame = mask_frame(frame_rgb)

        cv2.imshow("Video Stream", masked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    player.player.stop()
