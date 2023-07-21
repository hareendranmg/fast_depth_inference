from PIL import Image
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

# Load the image
image_path = "C:/Users/haree/OneDrive/Pictures/Camera Roll/WIN_20230721_18_57_03_Pro.jpg"
image = Image.open(image_path).convert("RGB")

# Resize the image to match the model's input size
image = image.resize((224, 224))

# Convert the image to a NumPy array and normalize pixel values
image_np = np.array(image) / 255.0

# Expand dimensions to create a batch of size 1 (required for inference)
input_data = np.expand_dims(image_np.transpose(2, 0, 1).astype(np.float32), axis=0)

# Load the ONNX model
model_path = "C:/Users/haree/Downloads/fast-depth (2).onnx"
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
plt.colorbar()
plt.title('Depth Map')
plt.axis('off')

plt.show()
