from PIL import Image
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt


# Load the image
image_path = "C:/Users/haree/python_dev/fast_depth_inference/rgb.png"
image = Image.open(image_path).convert("RGB")

# Resize the image to match the model's input size
image = image.resize((224, 224))

# Convert the image to a NumPy array and normalize pixel values
image_np = np.array(image) / 255.0

# Expand dimensions to create a batch of size 1 (required for inference)
# input_data = np.expand_dims(image_np.astype(np.float32), axis=0)
input_data = np.expand_dims(image_np.transpose(2, 0, 1).astype(np.float32), axis=0)



# Load the ONNX model
model_path = "C:/Users/haree/Downloads/fast-depth (2).onnx"
ort_session = onnxruntime.InferenceSession(model_path)

input_tensor_name = "input.1"
output_data = ort_session.run(None, {input_tensor_name: input_data})

output_data_str = np.array2string(output_data[0], separator=', ')

# print(output_data_str)
with open('example.txt', 'w') as file:
    # Append data to the file using the 'write()' method
    file.write(output_data_str)
