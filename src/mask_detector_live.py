import cv2
import numpy as np
from keras.models import load_model

# Paths
MODEL_PATH = r"C:\Users\91812\OneDrive\Desktop\face-mask-detector\models\mask_detector.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load model and face detector
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (so it feels like a mirror)
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # more accurate than 1.05
        minNeighbors=6,    # helps avoid false detections
        minSize=(80, 80)   # ignore very small detections
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            # Resize to model input size
            target_size = (150, 150)  # must match your model
            face_resized = cv2.resize(face, target_size)
            face_normalized = face_resized.astype("float32") / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)

            # Predict mask or no mask
            pred = model.predict(face_expanded, verbose=0)[0][0]

            # Adjust threshold if needed
            label = "Mask" if pred > 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print(f"Error processing face: {e}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Show output
    cv2.imshow("Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
