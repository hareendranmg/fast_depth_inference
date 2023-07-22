import os
import cv2
import numpy as np
import onnxruntime

# Load the ONNX model
model_path = 'fast-depth.onnx'
ort_session = onnxruntime.InferenceSession(model_path)
input_tensor_name = "input.1"

# Function to process video frames
def process_frame(frame):
    # Resize the frame to match the model's input size
    resized_frame = cv2.resize(frame, (224, 224))

    # Convert the frame to a NumPy array and normalize pixel values
    image_np = np.array(resized_frame) / 255.0

    # Expand dimensions to create a batch of size 1 (required for inference)
    input_data = np.expand_dims(image_np.transpose(2, 0, 1).astype(np.float32), axis=0)

    # Perform inference with the ONNX model
    output_data = ort_session.run(None, {input_tensor_name: input_data})
    depth_map = np.squeeze(output_data[0])

    # Return the processed frame
    return depth_map


# Create a VideoCapture object to capture frames from /dev/video2
cap = cv2.VideoCapture('/dev/video2')

# Run the processing loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        depth_map = process_frame(frame)

        # Convert the depth map to a color image (optional, only for visualization purposes)
        depth_map_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Display the original frame and the depth map side by side
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Depth Map', depth_map_colored)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Release the VideoCapture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Stop the GStreamer pipeline and clean up
pipeline.set_state(Gst.State.NULL)
