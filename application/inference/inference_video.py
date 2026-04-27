from ultralytics import YOLO
import cv2, os
import json
import time
from datetime import datetime, timedelta
from service.inference_db_service import get_detection_settings_db, update_job_info_db, create_detection_results_db
from .inference_utils import (
    calculate_roi_from_video, 
    draw_roi_lines, 
    check_apd, 
    save_person_crop, 
    overlay_info
)
import numpy as np

MODEL_PATH = r"models/best.pt"
OUTPUT_DIR = r"video_res"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_video_writer(cap, output_path):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def check_person_in_roi(frame, r, person_states, compilance_res, roi_top, roi_bottom, do_apd_check=True, current_dt=None):
    status = []
    if r.boxes is None: return status

    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([None] * len(boxes))
    model_names = r.names

    for i, box in enumerate(boxes):
        cls = int(classes[i]); label = str(model_names[cls])
        if label != "person": continue
        person_id = int(ids[i]) if ids[i] is not None else None
        px1, py1, px2, py2 = map(int, box)

        if person_id not in person_states: person_states[person_id] = {"bottom_touched": False}
        state = person_states[person_id]

        if py1 <= roi_bottom <= py2:
            if not state["bottom_touched"] and person_id in compilance_res:
                apd = compilance_res[person_id]
                if "image_data" not in apd:
                    missing = any(not apd.get(k, False) for k in ["apron", "gloves", "boots", "mask", "hairnet"])
                    if missing:
                        img_data = save_person_crop(person_id, box, frame, r)
                        if img_data:
                            compilance_res[person_id]["image_data"] = img_data

            state["bottom_touched"] = True
            roi_text = "no"
        elif not state["bottom_touched"] and roi_top <= py2:
            roi_text = "yes"
            if do_apd_check: check_apd(person_id, box, boxes, classes, compilance_res, model_names, current_dt=current_dt)
        else:
            roi_text = "no"

        status.append((person_id, roi_text))
    return status

def process_frame(frame, model, person_states, compilance_res, roi_top, roi_bottom, do_apd_check=True, current_dt=None):
    results = model.track(frame, persist=True, tracker="trackers/bytetrack.yaml", verbose=False)
    r = results[0]

    annotated_video = r.plot(line_width=2, conf=False, font_size=0.8)
    annotated_video = draw_roi_lines(annotated_video, roi_top, roi_bottom)

    status = check_person_in_roi(frame, r, person_states, compilance_res, roi_top, roi_bottom, do_apd_check, current_dt=current_dt)
    annotated_video = overlay_info(annotated_video, status)

    return annotated_video

def run_inference(video_path, save_video=True, metadata=None):
    model = YOLO(MODEL_PATH)
    model.verbose = False

    user_id = metadata.get("user_id") if metadata else None
    job_id = metadata.get("job_id") if metadata else None
    
    settings = get_detection_settings_db(int(user_id)) if user_id else {"top_roi": 25, "bottom_roi": 75, "frame_interval": 2}
    top_percent = settings["top_roi"]
    bottom_percent = settings["bottom_roi"]
    apd_every_n_frames = settings.get("frame_interval", 2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 1, 0, iter([])
        
    roi_top, roi_bottom = calculate_roi_from_video(cap, top_percent, bottom_percent)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    video_start_dt_str = metadata.get("data_datetime") if metadata else None
    video_start_dt = datetime.fromisoformat(video_start_dt_str) if video_start_dt_str else None

    def inference_generator():
        person_states = {}
        compilance_res = {}

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
                if (current_time - last_sent_time) >= 1.0 or frame_count == total_frames:
                    progress_data = {"status": "Running", "frame": frame_count, "total_frames": total_frames}
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    last_sent_time = current_time

            cap.release()
            if save_video and writer: writer.release()

            filtered_compilance_res = {}
            for pid, res in compilance_res.items():
                person = person_states.get(pid, {})
                if person.get("bottom_touched", False):
                    filtered_compilance_res[pid] = res

            compilance_res = filtered_compilance_res

            if job_id:
                update_job_info_db(job_id, status="Done", video_result_path=output_path)
                create_detection_results_db(job_id, compilance_res)

            final_data = {"status": "Done", "frame": frame_count, "total_frames": total_frames}
            yield f"data: {json.dumps(final_data)}\n\n"
        except Exception as e:
            if cap: cap.release()
            if save_video and writer: writer.release()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return total_frames, fps, duration, inference_generator()
