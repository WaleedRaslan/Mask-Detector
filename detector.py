"""
detector.py
============
منطق الكشف المشترك: تحميل النموذج، كشف الوجوه بواسطة MediaPipe،
التنبؤ بوجود/عدم وجود الكمامة، رسم النتائج على الإطار، وتسجيل كل
عملية كشف في ملف CSV يُستخدم بعد ذلك في صفحة الإحصائيات.
"""

import os
import csv
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st

# ==========================
# المسارات
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "MaskDetector.keras")
LOG_PATH = os.path.join(BASE_DIR, "detection_log.csv")

IMG_SIZE = (224, 224)

LABELS_AR = {
    "mask": "يوجد كمامة",
    "no_mask": "لا يوجد كمامة",
}

# ==========================
# تحميل النموذج و MediaPipe (مع كاش لعدم إعادة التحميل)
# ==========================


@st.cache_resource(show_spinner="جاري تحميل نموذج الكشف عن الكمامة...")
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7,
    )


# ==========================
# التسجيل (Logging) للإحصائيات
# ==========================


def ensure_log_file():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "source", "label", "confidence"])


def log_detection(source: str, label: str, confidence: float):
    ensure_log_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                source,
                label,
                round(float(confidence), 4),
            ]
        )


def clear_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    ensure_log_file()


# ==========================
# التنبؤ على وجه واحد
# ==========================


def predict_face(face_bgr: np.ndarray):
    """
    يأخذ صورة وجه بصيغة BGR ويرجع:
    (label_key, confidence, raw_score)
    label_key: "mask" أو "no_mask"
    confidence: نسبة الثقة بالتصنيف النهائي (0-1)
    raw_score: القيمة الخام الناتجة عن النموذج (sigmoid)
    """
    model = get_model()

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    # ملاحظة: لا نعمل قسمة /255 لأن النموذج يحتوي على طبقة Rescaling داخلية
    face_input = np.expand_dims(face_resized, axis=0)

    prediction = model.predict(face_input, verbose=0)
    score = float(prediction[0][0])

    if score >= 0.5:
        return "no_mask", score, score
    else:
        return "mask", 1.0 - score, score


# ==========================
# كشف الوجوه في إطار كامل + رسم + تسجيل
# ==========================


def detect_and_annotate(frame_bgr: np.ndarray, source: str = "camera", do_log: bool = True):
    """
    يكتشف كل الوجوه في الإطار، يصنّف كل وجه (كمامة / بدون كمامة)،
    يرسم المربعات والنصوص على الإطار، ويسجّل النتائج إذا طُلب ذلك.

    يرجع: (الإطار المعدل, قائمة تفاصيل كل كشف)
    """
    detector = get_face_detector()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    detections = []
    h, w, _ = frame_bgr.shape

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = min(int(bbox.width * w), w - x)
            height = min(int(bbox.height * h), h - y)

            if width <= 0 or height <= 0:
                continue

            face = frame_bgr[y : y + height, x : x + width]
            if face.size == 0:
                continue

            label_key, confidence, raw_score = predict_face(face)
            label_text = LABELS_AR[label_key]
            color = (0, 255, 0) if label_key == "mask" else (0, 0, 255)

            cv2.rectangle(frame_bgr, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame_bgr,
                f"{'Mask' if label_key == 'mask' else 'No Mask'} {confidence * 100:.1f}%",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            detections.append(
                {
                    "bbox": (x, y, width, height),
                    "label": label_key,
                    "label_text": label_text,
                    "confidence": confidence,
                    "raw_score": raw_score,
                }
            )

            if do_log:
                log_detection(source, label_key, confidence)

    return frame_bgr, detections
