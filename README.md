# Capturing-image
#Image will be captured through WebCam and identify the objects in the image
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def capture_image():
    # Initialize the webcam (use 0 for the default camera, 1 or higher for other connected cameras)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return None

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if ret:
        # Save the captured image
        img_name = "captured_image.png"
        cv2.imwrite(img_name, frame)
        print(f"Image saved as {img_name}")

        # Release the camera
        cap.release()

        return img_name
    else:
        print("Error: Could not read frame from webcam.")
        # Release the camera if it was opened
        if cap.isOpened():
            cap.release()
        return None

def analyze_image(image_path):
    # Load the image and preprocess it
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return None

    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to fit the model input size
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Preprocess the image
    img_preprocessed = preprocess_input(img_resized)

    # Add a batch dimension
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Load the pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet')

    # Predict the content of the image
    predictions = model.predict(img_batch)
    decoded_predictions = decode_predictions(predictions, top=10)[0]  # Return top 10 predictions

    return decoded_predictions


def main():
    # Capture an image from the webcam
    image_path = capture_image()

    if image_path:
        # Analyze the captured image
        predictions = analyze_image(image_path)

        if predictions:
            print("Predictions:")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                print(f"{i + 1}. {label} ({score * 100:.2f}%)")

            # Interactive Q&A based on predictions
            while True:
                user_input = input("\nAsk a question about the image or type 'exit' to quit: ").strip().lower()

                if user_input == 'exit':
                    break

                found_match = False
                for _, label, score in predictions:
                    if label.lower() in user_input:
                        print(f"Yes, there is a {label} in the image with a confidence of {score * 100:.2f}%.")
                        found_match = True
                        break

                if not found_match:
                    print("Sorry, I couldn't find anything related to your question in the image.")

        print("Goodbye!")

if __name__ == "__main__":
    main()
