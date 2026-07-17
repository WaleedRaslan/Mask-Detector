import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("MaskDetector.keras")

# ==========================
# MediaPipe
# ==========================
mp_face = mp.solutions.face_detection

face_detection = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
)

# ==========================
# Camera
# ==========================
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)

            width = min(width, w - x)
            height = min(height, h - y)

            face = frame[y:y+height, x:x+width]

            if face.size == 0:
                continue

            # ==========================
            # Preprocessing
            # ==========================

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))

            # لا تعمل /255 لأن النموذج يحتوي على Rescaling
            face = np.expand_dims(face, axis=0)

            # ==========================
            # Prediction
            # ==========================

            prediction = model.predict(face, verbose=0)

            score = prediction[0][0]

            print(score)

            if score >= 0.5:
                label = "No Mask"
                color = (0, 0, 255)
                confidence = score
            else:
                label = "Mask"
                color = (0, 255, 0)
                confidence = 1 - score

            # ==========================
            # Draw
            # ==========================

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.imshow("Face", face[0])

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()