import os
from PIL import Image
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import image_display_module


# Path to the image file you want to display
image_path = os.path.join(os.getcwd(),'rgb.png')
image = Image.open(image_path).convert("RGB")

# Resize the image to match the model's input size
image = image.resize((224, 224))

# Convert the image to a NumPy array and normalize pixel values
image_np = np.array(image) / 255.0

# Expand dimensions to create a batch of size 1 (required for inference)
input_data = np.expand_dims(image_np.transpose(2, 0, 1).astype(np.float32), axis=0)

# Load the ONNX model
model_path = 'fast-depth.onnx'

ort_session = onnxruntime.InferenceSession(model_path)

input_tensor_name = "input.1"
output_data = ort_session.run(None, {input_tensor_name: input_data})

# Access the output tensor with name "424"
depth_map = output_data[0]
depth_map = np.squeeze(depth_map)


# Display the original image and the depth map side by side
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(depth_map, cmap='viridis')
# plt.colorbar()
plt.title('Depth Map')
plt.axis('off')

# Save the plot as an image file
plt.savefig("depth_map.png")
plt.close()

depth_map_image_path = os.path.join(os.getcwd(),'depth_map.png')

# Display the image using GStreamer
image_display_module.display_image(depth_map_image_path)