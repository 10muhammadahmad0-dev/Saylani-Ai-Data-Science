
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import tempfile
import os

app = FastAPI(title="Pothole Detection API")

# Load YOLO model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo_model.pt"

model = YOLO(str(MODEL_PATH))

CONFIDENCE = 0.25


@app.get("/")
def home():
    return {
        "message": "Pothole Detection API is running"
    }


# IMAGE DETECTION
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "box": [
                round(float(x), 2)
                for x in box.xyxy[0].tolist()
            ]
        })

    annotated_image = result.plot()

    success, buffer = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Image processing failed"
        )

    import base64

    encoded_image = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return {
        "total_potholes": len(detections),
        "detections": detections,
        "image": encoded_image
    }


# VIDEO DETECTION
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    allowed_extensions = [
        ".mp4", ".avi", ".mov", ".mkv"
    ]

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    input_path = None
    output_path = None
    cap = None
    writer = None

    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:
            temp.write(await file.read())
            input_path = temp.name

        output_path = input_path + "_processed.mp4"

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Cannot open uploaded video"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 25

        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid video dimensions"
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Could not create output video"
            )

        total_frames = 0
        total_potholes = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )

            result = results[0]

            total_potholes += len(result.boxes)
            total_frames += 1

            annotated_frame = result.plot()

            writer.write(annotated_frame)

        cap.release()
        cap = None

        writer.release()
        writer = None

        if total_frames == 0 or not os.path.exists(output_path):
            raise HTTPException(
                status_code=400,
                detail="No frames processed"
            )

        with open(output_path, "rb") as video:
            video_data = video.read()

        return Response(
            content=video_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition":
                    'attachment; filename="pothole_detection.mp4"',
                "X-Total-Frames": str(total_frames),
                "X-Total-Detections": str(total_potholes)
            }
        )

    finally:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        if output_path and os.path.exists(output_path):
            os.remove(output_path)
