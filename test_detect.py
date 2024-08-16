from PIL import Image
import numpy as np
import cv2
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

def mask_frame(image, model_path, keep_aspect_ratio=False):
    """Apply semantic segmentation on the input image and overlay the mask."""
    # Load the model
    interpreter = make_interpreter(model_path, device=':0')
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)

    # Resize the input image
    img = Image.fromarray(image)
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
    mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

    # Convert mask to RGBA and add transparency
    mask_img = mask_img.convert("RGBA")
    datas = mask_img.getdata()

    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((255, 255, 255, 0))  # Fully transparent
        else:
            new_data.append((item[0], item[1], item[2], 128))  # Set transparency

    mask_img.putdata(new_data)

    # Convert the resized input image to RGBA
    base_img = resized_img.convert("RGBA")

    # Overlay the mask on the base image
    combined = Image.alpha_composite(base_img, mask_img)

    # Convert back to RGB
    combined = combined.convert("RGB")

    # Convert to numpy array and return the result
    return np.array(combined)

# Example usage
if __name__ == '__main__':
    input_image = np.array(Image.open('bird.bmp'))
    model_path = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'

    output_image = mask_frame(input_image, model_path, keep_aspect_ratio=True)
    cv2.imwrite('output.jpg', output_image)