# src/inference/face_detector.py

import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def extract_face(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Largest detected face
    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest_face

    # Add some padding
    padding = 20

    x = max(0, x - padding)
    y = max(0, y - padding)

    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    face = image[
        y:y+h,
        x:x+w
    ]

    return face