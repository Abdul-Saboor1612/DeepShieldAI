# src/inference/frame_extractor.py

import cv2
import os


def extract_frames(
    video_path,
    output_dir,
    frame_interval=30
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for file in os.listdir(output_dir):
        if file.endswith(".jpg"):
            os.remove(
                os.path.join(
                    output_dir,
                    file
            )
        )

    cap = cv2.VideoCapture(
        video_path
    )

    frame_count = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:

            frame_path = os.path.join(
                output_dir,
                f"frame_{saved_count}.jpg"
            )

            cv2.imwrite(
                frame_path,
                frame
            )

            saved_count += 1

        frame_count += 1

    cap.release()

    return saved_count