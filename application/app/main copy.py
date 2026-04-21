from fastapi import FastAPI
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_PATH = r"D:\AIPROJECT\TA\backend2\application\models\best.pt"
VIDEO_PATH = r"D:\AIPROJECT\TA\backend2\application\videos\video_test_3.mp4"
OUTPUT_DIR = r"D:\AIPROJECT\TA\backend2\application\video_res"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI_TOP = 800
ROI_BOTTOM = 1000

app = FastAPI()

# Load YOLO model
model = YOLO(MODEL_PATH)
model.verbose = False

# ==============================
# STATE
# ==============================
person_states = {}     
compilance_res = {}    

# ==============================
# UTILS
# ==============================
def create_video_writer(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

def draw_roi_lines(frame):
    h, w = frame.shape[:2]
    color = (0, 0, 255)
    thickness = 2
    cv2.line(frame, (0, ROI_TOP), (w, ROI_TOP), color, thickness)
    cv2.line(frame, (0, ROI_BOTTOM), (w, ROI_BOTTOM), color, thickness)
    cv2.putText(frame, "ROI TOP", (10, ROI_TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "ROI BOTTOM", (10, ROI_BOTTOM - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ==============================
# APD CHECK LOGIC
# ==============================
def check_apd(person_id, person_box, frame_boxes, frame_classes, overlap_thresh=0.5):
    global compilance_res
    if person_id is not None:
        person_id = int(person_id)

    if person_id not in compilance_res:
        compilance_res[person_id] = {
            "apron": False,
            "gloves": False,
            "boots": False,
            "mask": False,
            "hairnet": False
        }

    x1_person, y1_person, x2_person, y2_person = map(int, person_box)

    for i, box in enumerate(frame_boxes):
        cls = int(frame_classes[i])
        label = str(model.names[cls])
        if label == "person":
            continue

        x1_obj, y1_obj, x2_obj, y2_obj = map(int, box)
        xA = max(x1_person, x1_obj)
        yA = max(y1_person, y1_obj)
        xB = min(x2_person, x2_obj)
        yB = min(y2_person, y2_obj)

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        obj_area = max(1, (x2_obj - x1_obj) * (y2_obj - y1_obj))

        if inter_area / obj_area >= overlap_thresh:
            compilance_res[person_id][label] = True

# ==============================
# APD CHECK PER INTERVAL FRAME
# ==============================
def should_do_apd_check(frame_count, interval_frames):
    """
    Cek APD tiap interval tertentu dalam jumlah frame
    """
    return frame_count % interval_frames == 0

# ==============================
# PERSON ROI CHECK
# ==============================
def check_person_in_roi_stateful(boxes, classes, ids, do_apd_check=True):
    global person_states
    person_status = []

    if boxes is None or len(boxes) == 0:
        return person_status

    for i, box in enumerate(boxes):
        cls = int(classes[i])
        label = str(model.names[cls])
        if label != "person":
            continue

        person_id = int(ids[i]) if ids[i] is not None else None
        x1, y1, x2, y2 = map(int, box)

        if person_id not in person_states:
            person_states[person_id] = {"bottom_touched": False}

        state = person_states[person_id]

        if y1 <= ROI_BOTTOM <= y2:
            state["bottom_touched"] = True
            roi_text = "no"
        elif not state["bottom_touched"] and y1 <= ROI_TOP <= y2:
            roi_text = "yes"
            if do_apd_check:
                check_apd(person_id, box, boxes, classes)
        else:
            roi_text = "no"

        person_status.append((person_id, roi_text))
    return person_status

# ==============================
# OVERLAY INFO
# ==============================
def overlay_person_info(frame, person_status):
    if not person_status:
        return frame
    start_y = 30
    line_height = 25
    for idx, (person_id, roi_text) in enumerate(person_status):
        display_text = f"Person {person_id}: {roi_text}" if person_id is not None else f"Person: {roi_text}"
        color = (0, 255, 0) if roi_text == "yes" else (0, 0, 255)
        y = start_y + idx * line_height
        cv2.putText(frame, display_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# ==============================
# FRAME PROCESSING
# ==============================
def process_frame(frame, do_apd_check=True):
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    r = results[0]

    annotated_frame = r.plot(line_width=2, conf=False, font_size=0.8)
    annotated_frame = draw_roi_lines(annotated_frame)

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else np.array([])
    classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.array([])
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([None]*len(boxes))

    person_status = check_person_in_roi_stateful(boxes, classes, ids, do_apd_check)
    annotated_frame = overlay_person_info(annotated_frame, person_status)
    return annotated_frame

# ==============================
# INFERENCE PIPELINE
# ==============================
def run_inference(video_path, save_video=True, apd_interval_frames=15):
    """
    apd_interval_frames: cek APD tiap N frame
    """
    global person_states, compilance_res
    person_states = {}
    compilance_res = {}

    cap = cv2.VideoCapture(video_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4") if save_video else None
    writer = create_video_writer(cap, output_path) if save_video else None

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        do_apd_check = should_do_apd_check(frame_count, apd_interval_frames)
        annotated_frame = process_frame(frame, do_apd_check)

        if save_video and writer:
            writer.write(annotated_frame)

        frame_count += 1

    cap.release()
    if save_video and writer:
        writer.release()

    return {
        "frames_processed": frame_count,
        "output_video": output_path,
        "compilance_res": compilance_res
    }

# ==============================
# FASTAPI ENDPOINT
# ==============================
@app.get("/run-inference")
def run_model(save_video: bool = True, apd_interval_frames: int = 5):
    """
    save_video: True/False untuk output video
    apd_interval_frames: cek APD tiap N frame
    """
    result = run_inference(VIDEO_PATH, save_video=save_video, apd_interval_frames=apd_interval_frames)
    return {"status": "Done", "result": result}