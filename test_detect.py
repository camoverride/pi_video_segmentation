import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label."""
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def mask_frame(frame, interpreter, keep_aspect_ratio=False):
    """Apply semantic segmentation on the input frame and overlay the mask."""
    width, height = common.input_size(interpreter)

    # Resize the input frame
    img = Image.fromarray(frame)
    if keep_aspect_ratio:
        resized_img, _ = common.set_resized_input(
            interpreter, img.size, lambda size: img.resize(size, Image.LANCZOS))
    else:
        resized_img = img.resize((width, height), Image.LANCZOS)
        common.set_input(interpreter, resized_img)

    # Run inference
    interpreter.invoke()

    # Get the segmentation result
    result = segment.get_output(interpreter)
    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

    # If keep_aspect_ratio, we need to remove the padding area.
    new_width, new_height = resized_img.size
    result = result[:new_height, :new_width]

    # Convert the mask to RGB and set the desired color
    mask_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    mask_img[result != 0] = [255, 0, 0]  # Set to red where the mask is present

    # Convert the mask to RGBA and apply transparency
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2RGBA)
    mask_img[:, :, 3] = 128  # Set transparency to 50%

    # Convert the resized input image to RGBA
    base_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2RGBA)

    # Overlay the mask on the base image
    combined = cv2.addWeighted(base_img, 1.0, mask_img, 0.5, 0)

    # Convert back to RGB
    combined = cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB)

    # Return the result as a numpy array
    return combined

def run_webcam_segmentation(model_path):
    """Capture frames from the webcam, apply the mask, and display the output."""
    # Load the model
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Apply mask to the current frame
        masked_frame = mask_frame(frame, interpreter)

        # Display the resulting frame
        cv2.imshow('Semantic Segmentation', masked_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
    run_webcam_segmentation(model_path)
