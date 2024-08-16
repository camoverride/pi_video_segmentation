import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, segment

# Load the segmentation model
model_path = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load the input image
input_image_path = 'cam_rgb.jpg'
image = Image.open(input_image_path)
width, height = image.size
_, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size))

# Run inference
interpreter.invoke()
result = segment.get_output(interpreter)

# Resize the segmentation result to match the original image size
result_resized = Image.fromarray(result.astype(np.uint8)).resize((width, height), Image.NEAREST)
result_resized = np.array(result_resized)

# Create a color map for segmentation
colors = {
    0: (255, 0, 0, 128),    # Person in translucent red
    18: (255, 255, 0, 128), # Dog in translucent yellow
    2: (0, 255, 255, 128)   # Bicycle in translucent cyan
}

# Load a font for labels
font = ImageFont.load_default()

# Create a transparent layer for overlay
overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Apply the segmentation masks
for y in range(height):
    for x in range(width):
        label = result_resized[y, x]
        if label in colors:
            # Draw the translucent mask
            draw.point((x, y), fill=colors[label])

# Composite the original image with the overlay
output_image = Image.alpha_composite(image.convert("RGBA"), overlay)

# Save the output image
output_image_path = 'segmentation_result.png'
output_image.save(output_image_path)

print(f"Segmentation result saved at {output_image_path}")
