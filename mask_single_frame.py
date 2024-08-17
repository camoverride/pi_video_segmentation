import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

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

    # Create a 4-channel image with the mask applied
    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    rgba_frame[:, :, 3] = mask_resized

    # Set background to black or transparent where mask is not applied
    for c in range(3):
        rgba_frame[:, :, c] = rgba_frame[:, :, c] * (mask_resized // 255)

    return rgba_frame

def capture_and_segment(model_path):
    """
    Capture a single frame from the webcam, apply the mask, and display the output.
    """
    # Load the model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture one frame from the webcam
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image.")
        return

    # Apply mask to the captured frame (keeping only "person")
    masked_frame = mask_frame(frame, interpreter, selected_label="person")

    # Display the resulting image
    cv2.imshow('Segmented Image', masked_frame)

    # Press any key to close the window
    cv2.waitKey("q")
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite('masked_image.png', masked_frame)

if __name__ == '__main__':
    model_path = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
    capture_and_segment(model_path)
