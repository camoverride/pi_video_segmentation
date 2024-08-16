import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.adapters.segment import get_output

# Load the segmentation model
model_path = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load the input image
input_image_path = '/mnt/data/Screenshot 2024-08-15 at 7.30.36 PM.png'
image = Image.open(input_image_path)
_, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size))

# Run inference
interpreter.invoke()
result = get_output(interpreter)

# Create a color map for segmentation
colors = {
    1: (255, 0, 0, 128),    # Person in translucent red
    18: (255, 255, 0, 128), # Dog in translucent yellow
    2: (0, 255, 255, 128)   # Bicycle in translucent cyan
}

# Load a font for labels
font = ImageFont.load_default()

# Create a transparent layer for overlay
overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

# Apply the segmentation masks
draw = ImageDraw.Draw(overlay)
for obj in result:
    label = obj.id
    if label in colors:
        # Draw the translucent mask
        draw.rectangle(obj.bbox, fill=colors[label])
        # Draw the category label
        draw.text((obj.bbox[0], obj.bbox[1] - 10), str(label), fill=(255, 255, 255, 255), font=font)

# Composite the original image with the overlay
output_image = Image.alpha_composite(image.convert("RGBA"), overlay)

# Save the output image
output_image_path = '/mnt/data/segmentation_result.png'
output_image.save(output_image_path)

print(f"Segmentation result saved at {output_image_path}")
