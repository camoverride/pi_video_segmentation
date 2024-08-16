import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from screeninfo import get_monitors



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


def label_to_color_image(label):
    """
    Adds color defined by the dataset colormap to the label.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def mask_frame(frame, interpreter, selected_labels, keep_aspect_ratio=False):
    """
    Apply semantic segmentation on the input frame, filter by selected labels,
    size up the mask to match the original frame, and overlay the mask.
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

    # Filter the segmentation result to only include selected labels
    filtered_result = np.zeros_like(result)
    for label_name in selected_labels:
        label_index = LABELS.index(label_name)
        filtered_result[result == label_index] = label_index

    # Size up the mask to match the original frame dimensions
    mask_img = cv2.resize(filtered_result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    color_mask = create_pascal_label_colormap()[mask_img]

    # Convert the mask to RGBA and apply transparency
    color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGB2RGBA)
    color_mask[:, :, 3] = 128  # Set transparency to 50%

    # Convert the original input frame to RGBA
    base_img = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    # Overlay the mask on the original frame
    combined = cv2.addWeighted(base_img, 1.0, color_mask, 0.5, 0)

    # Convert back to RGB
    combined = cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB)

    # Add labels
    labeled_frame = add_labels(combined, mask_img)

    # Return the result as a numpy array
    return labeled_frame


def add_labels(image, segmentation_result):
    """
    Add labels to the detected objects on the image.
    """
    unique_labels = np.unique(segmentation_result)

    for label in unique_labels:
        if label == 0:  # Skip the background
            continue
        
        # Get the mask for the current label
        mask = segmentation_result == label
        
        # Calculate the centroid of the mask
        y_coords, x_coords = np.where(mask)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        # Draw the label on the image
        label_name = LABELS[label]
        cv2.putText(image, label_name, (centroid_x, centroid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image


def run_webcam_segmentation(model_path, categories):
    """
    Capture frames from the webcam, apply the mask, and display the output.
    """
    # Convert the categories to their corresponding label indices
    selected_labels = [label for label in categories if label in LABELS]

    # Load the model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get the screen resolution using screeninfo
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    # Create a named window in fullscreen mode
    cv2.namedWindow("Semantic Segmentation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Semantic Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Apply mask to the current frame
        masked_frame = mask_frame(frame, interpreter, selected_labels)

        # Resize the output frame to fill the screen
        resized_frame = cv2.resize(masked_frame, (screen_width, screen_height))

        # Display the resulting frame
        cv2.imshow('Semantic Segmentation', resized_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
    categories = ["cat", "dog", "person"]

    run_webcam_segmentation(model_path, categories)
