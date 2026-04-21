from ultralytics import YOLO
import cv2, os
import json
import time
import numpy as np
from datetime import datetime, timedelta
import base64
from service.inference_db_service import get_detection_settings_db, update_job_info_db, create_detection_results_db

# Load settings once
MODEL_PATH = r"models/best.pt"
OUTPUT_DIR = r"video_res"
IMAGE_OUTPUT_DIR = r"tmp_image_res"
# ROI settings will be loaded per inference

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# model = YOLO(MODEL_PATH) - Moved into run_inference for parallel safety
# model.verbose = False


def calculate_roi_from_video(cap, top_percent, bottom_percent):
    """
    Calculate ROI TOP and BOTTOM based on video dimensions and percentage.
    """
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_top = int(height * (top_percent / 100))
    roi_bottom = int(height * (bottom_percent / 100))
    return roi_top, roi_bottom

def create_video_writer(cap, output_path):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def draw_roi_lines(frame, roi_top, roi_bottom):
    h, w = frame.shape[:2]
    color = (0, 0, 255)
    cv2.line(frame, (0, roi_top), (w, roi_top), color, 2)
    cv2.line(frame, (0, roi_bottom), (w, roi_bottom), color, 2)
    cv2.putText(frame, "ROI TOP", (10, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "ROI BOTTOM", (10, roi_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def check_apd(person_id, person_box, frame_boxes, frame_classes, compilance_res, model_names, overlap_thresh=0.5, current_dt=None):
    if person_id is None: return
    person_id = int(person_id)
    if person_id not in compilance_res:
        compilance_res[person_id] = {
            "apron": False, 
            "gloves": False, 
            "boots": False, 
            "mask": False, 
            "hairnet": False,
            "detection_time": current_dt
        }

    x1, y1, x2, y2 = map(int, person_box)
    for i, box in enumerate(frame_boxes):
        cls = int(frame_classes[i])
        label = str(model_names[cls])
        if label == "person": continue

        x1o, y1o, x2o, y2o = map(int, box)
        xA = max(x1, x1o); yA = max(y1, y1o)
        xB = min(x2, x2o); yB = min(y2, y2o)
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        obj_area = max(1, (x2o - x1o) * (y2o - y1o))
        if inter_area / obj_area >= overlap_thresh:
            compilance_res[person_id][label] = True

def save_person_crop(person_id, box, frame):
    """
    Save person crop image to IMAGE_OUTPUT_DIR and return (path, binary_data).
    """
    img_name = f"person_{person_id}_{int(time.time())}.jpg"
    img_path = os.path.join(IMAGE_OUTPUT_DIR, img_name)
    
    x1, y1, x2, y2 = map(int, box)
    padding = 10
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        cv2.imwrite(img_path, crop)
        # Encode to binary for DB storage
        success, buffer = cv2.imencode(".jpg", crop)
        if success:
            return buffer.tobytes()
    return None

def check_person_in_roi(frame, boxes, classes, ids, person_states, compilance_res, roi_top, roi_bottom, model_names, do_apd_check=True, current_dt=None):
    status = []
    if boxes is None or len(boxes) == 0: return status

    for i, box in enumerate(boxes):
        cls = int(classes[i]); label = str(model_names[cls])
        if label != "person": continue
        person_id = int(ids[i]) if ids[i] is not None else None
        x1, y1, x2, y2 = map(int, box)

        if person_id not in person_states: person_states[person_id] = {"bottom_touched": False}
        state = person_states[person_id]


        if y1 <= roi_bottom <= y2:
            # Check if this is the transition to bottom_touched
            if not state["bottom_touched"] and person_id in compilance_res:
                apd = compilance_res[person_id]
                # Check if any APD is missing and no image saved yet
                if "image_data" not in apd:
                    missing = any(not apd.get(k, False) for k in ["apron", "gloves", "boots", "mask", "hairnet"])
                    if missing:
                        img_data = save_person_crop(person_id, box, frame)
                        if img_data:
                            compilance_res[person_id]["image_data"] = img_data

            state["bottom_touched"] = True
            roi_text = "no"
        elif not state["bottom_touched"] and roi_top <= y2:
            roi_text = "yes"
            if do_apd_check: check_apd(person_id, box, boxes, classes, compilance_res, model_names, current_dt=current_dt)
        else:
            roi_text = "no"

        # if y1 <= roi_bottom <= y2 and not state["top_touched"]:
        #     state["bottom_touched"] = True
        #     roi_text = "no"
        # elif y1 >= roi_bottom and state["top_touched"]:
        #     state["bottom_touched"] = True
        #     roi_text = "no"
        # elif not state["bottom_touched"] and y1 <= roi_top <= y2:
        #     state["top_touched"] = True
        #     roi_text = "yes"
        #     if do_apd_check: check_apd(person_id, box, boxes, classes, compilance_res)
        # else:
        #     roi_text = "no"
        status.append((person_id, roi_text))
    return status

def overlay_info(frame, person_status):
    start_y = 30; line_height = 25
    for idx, (pid, roi_text) in enumerate(person_status):
        txt = f"Person {pid}: {roi_text}" if pid is not None else f"Person: {roi_text}"
        color = (0, 255, 0) if roi_text == "yes" else (0, 0, 255)
        y = start_y + idx * line_height
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def process_frame(frame, model, person_states, compilance_res, roi_top, roi_bottom, do_apd_check=True, current_dt=None):
    results = model.track(frame, persist=True, tracker="trackers/bytetrack.yaml", verbose=False)
    r = results[0]

    annotated = r.plot(line_width=2, conf=False, font_size=0.8)
    annotated = draw_roi_lines(annotated, roi_top, roi_bottom)

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else np.array([])
    classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.array([])
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([None] * len(boxes))
    model_names = model.names

    status = check_person_in_roi(frame, boxes, classes, ids, person_states, compilance_res, roi_top, roi_bottom, model_names, do_apd_check, current_dt=current_dt)
    annotated = overlay_info(annotated, status)

    return annotated

def run_inference(video_path, save_video=True, metadata=None):
    # Initialize separate model per request for parallel safety
    model = YOLO(MODEL_PATH)
    model.verbose = False

    # Load settings from DB for this user
    user_id = metadata.get("user_id") if metadata else None
    job_id = metadata.get("job_id") if metadata else None
    
    settings = get_detection_settings_db(int(user_id)) if user_id else {"top_roi": 25, "bottom_roi": 75}
    top_percent = settings["top_roi"]
    bottom_percent = settings["bottom_roi"]
    apd_every_n_frames = settings["frame_interval"]

    # Open video for metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, iter([])
        
    roi_top, roi_bottom = calculate_roi_from_video(cap, top_percent, bottom_percent)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    video_start_dt_str = metadata.get("video_datetime") if metadata else None
    video_start_dt = datetime.fromisoformat(video_start_dt_str) if video_start_dt_str else None

    def inference_generator():
        person_states = {}
        compilance_res = {}

        # Video output
        output_path = (
            os.path.join(OUTPUT_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            if save_video else None
        )
        writer = create_video_writer(cap, output_path) if save_video else None

        frame_count = 0
        last_sent_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_dt = (video_start_dt + timedelta(seconds=frame_count / fps)).replace(microsecond=0) if video_start_dt else None
                do_apd_check = (frame_count % apd_every_n_frames == 0)
                annotated = process_frame(frame, model, person_states, compilance_res, roi_top, roi_bottom, do_apd_check, current_dt=current_dt)

                if save_video and writer:
                    writer.write(annotated)

                frame_count += 1

                current_time = time.time()
                # Yield progress SSE every 1 second
                if (current_time - last_sent_time) >= 1.0 or frame_count == total_frames:
                    progress_data = {
                        "status": "Running",
                        "frame": frame_count,
                        "total_frames": total_frames
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    last_sent_time = current_time

            cap.release()
            if save_video and writer:
                writer.release()

            # Only save t db when person touched bottom roi
            filtered_compilance_res = {}

            for pid, res in compilance_res.items():
                person = person_states.get(pid, {})
                bottom_touched = person.get("bottom_touched", False)

                if bottom_touched:
                    filtered_compilance_res[pid] = res

            compilance_res = filtered_compilance_res

            # Save results to DB instead of file
            if job_id:
                update_job_info_db(job_id, status="Done", video_result_path=output_path)
                create_detection_results_db(job_id, compilance_res)
                    

            # Final SSE data for front-end
            final_data = {
                "status": "Done",
                "frame": frame_count,
                "total_frames": total_frames,
            }
            yield f"data: {json.dumps(final_data)}\n\n"
        except Exception as e:
            if cap: cap.release()
            if save_video and writer: writer.release()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return total_frames, fps, duration, inference_generator()
