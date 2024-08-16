import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, segment

# Load the segmentation model
model_path = 'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
model_path = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load the input image
input_image_path = 'cam_rgb.jpg'
output_image_path = 'segmentation_result.jpg'
image = Image.open(input_image_path)
width, height = image.size
_, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size))

# Run inference
interpreter.invoke()
segmentation = segment.get_output(interpreter)

# Resize the segmentation result to match the original image size
segmentation_resized = Image.fromarray(segmentation.astype(np.uint8)).resize((width, height), Image.NEAREST)
segmentation_resized = np.array(segmentation_resized)

# Create a color map for segmentation (using simple RGB colors for simplicity)
colors = {
    15: (0, 255, 0, 128),    # Example class ID for "person"
    7: (255, 0, 0, 128),     # Example class ID for "road"
    14: (0, 0, 255, 128),    # Example class ID for "car"
}

# Prepare a transparent overlay
overlay = np.zeros((height, width, 4), dtype=np.uint8)

# Apply the mask using NumPy for better performance
for label, color in colors.items():
    mask = segmentation_resized == label
    overlay[mask] = color

# Convert overlay to an image and composite with the original
overlay_img = Image.fromarray(overlay, mode="RGBA")
output_image = Image.alpha_composite(image.convert("RGBA"), overlay_img)

# Save the output image
output_image.save(output_image_path)
print(f"Segmentation result saved at {output_image_path}")
