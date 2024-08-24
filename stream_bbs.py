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

    # Scale the bounding boxes back to the original frame size
    scale_x = frame.shape[1] / input_width
    scale_y = frame.shape[0] / input_height

    # Draw bounding boxes on the original frame
    for obj in boxes:
        ymin, xmin, ymax, xmax = obj.bbox
        ymin = int(ymin * scale_y)
        xmin = int(xmin * scale_x)
        ymax = int(ymax * scale_y)
        xmax = int(xmax * scale_x)

        # Adjust if the image is flipped or mirrored
        # Uncomment one of these if your boxes appear mirrored or flipped
        # xmin, xmax = frame.shape[1] - xmax, frame.shape[1] - xmin  # Horizontal flip
        # ymin, ymax = frame.shape[0] - ymax, frame.shape[0] - ymin  # Vertical flip

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f'{obj.id}: {obj.score:.2f}'
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
