import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter



# Initialize the Coral Edge TPU with the EfficientDet model
model_path = 'efficientdet-lite0_edgetpu.tflite'
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

if __name__ == "__main__":
    url = "https://61e0c5d388c2e.streamlock.net/live/MLK_E_Cherry_NS.stream/chunklist_w1373546751.m3u8"
    player = VLCPlayer(url)
    player.start()

    while True:
        frame = player.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        detected_frame = detect_objects(frame_rgb)

        cv2.imshow("Video Stream", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    player.player.stop()
