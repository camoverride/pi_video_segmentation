import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.adapters import classify
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file



# Load the TFLite model for the Coral TPU
model_path = "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size of the model
    input_size = common.input_size(interpreter)
    resized_frame = cv2.resize(frame, input_size)

    # Run the segmentation model on the frame
    common.set_input(interpreter, resized_frame)
    interpreter.invoke()
    segmented_output = segment.get_output(interpreter)

    # Resize the segmented output to match the original frame size
    segmented_output = cv2.resize(segmented_output, (frame.shape[1], frame.shape[0]))

    # Apply a color map to the segmented output for visualization
    color_segmented = cv2.applyColorMap(segmented_output, cv2.COLORMAP_JET)

    # Overlay the segmented output on the original frame
    overlay = cv2.addWeighted(frame, 0.7, color_segmented, 0.3, 0)

    # Display the frame with the overlay
    cv2.imshow('Segmented Frame', overlay)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
