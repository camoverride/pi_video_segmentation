from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from PIL import Image

# Load the model
model_path = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load an image
image_path = 'cam_rgb.jpg'  # Replace with your image path
image = Image.open(image_path)
_, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size))

# Run inference
interpreter.invoke()
objects = detect.get_objects(interpreter, score_threshold=0.5, image_scale=scale)

# Print detected objects
for obj in objects:
    if obj.id == 0:  # '0' corresponds to the 'person' class in the COCO dataset
        print('Person detected with score:', obj.score)