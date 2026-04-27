from ultralytics import YOLO
import cv2, os
import json
from datetime import datetime
from service.inference_db_service import get_detection_settings_db, update_job_info_db, create_detection_results_db
from .inference_utils import (
    check_apd, 
    save_person_crop
)

MODEL_PATH = r"models/best.pt"

def run_inference_images(image_paths, metadata=None):
    model = YOLO(MODEL_PATH)
    model.verbose = False

    user_id = metadata.get("user_id") if metadata else None
    job_id = metadata.get("job_id") if metadata else None
    
    settings = get_detection_settings_db(int(user_id)) if user_id else {"top_roi": 25, "bottom_roi": 75}
    top_percent = settings["top_roi"]
    bottom_percent = settings["bottom_roi"]

    total_frames = len(image_paths)
    video_start_dt_str = metadata.get("data_datetime") if metadata else None
    video_start_dt = datetime.fromisoformat(video_start_dt_str) if video_start_dt_str else None
    current_dt = video_start_dt.replace(microsecond=0) if video_start_dt else datetime.now().replace(microsecond=0)

    def inference_generator():
        try:
            compilance_res = {}
            global_person_counter = 0

            for idx, img_path in enumerate(image_paths):
                frame = cv2.imread(img_path)
                if frame is None: continue

                h, w = frame.shape[:2]
                roi_top = int(h * (top_percent / 100))
                roi_bottom = int(h * (bottom_percent / 100))

                results = model(frame, verbose=False)
                r = results[0]
                
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                model_names = r.names
                
                for i, box in enumerate(boxes):
                    cls = int(classes[i]); label = str(model_names[cls])
                    if label != "person": continue
                    
                    person_id = global_person_counter
                    global_person_counter += 1
                    px1, py1, px2, py2 = map(int, box)
                    
                    if py2 >= roi_top and py1 <= roi_bottom:
                        compilance_res[person_id] = {
                            "apron": False, "gloves": False, "boots": False, 
                            "mask": False, "hairnet": False, "detection_time": current_dt
                        }
                        
                        check_apd(person_id, box, boxes, classes, compilance_res, model_names, current_dt=current_dt)
                        
                        missing = any(not compilance_res[person_id].get(k, False) for k in ["apron", "gloves", "boots", "mask", "hairnet"])
                        if missing:
                            img_data = save_person_crop(person_id, box, frame, r)
                            if img_data:
                                compilance_res[person_id]["image_data"] = img_data
                
                progress_data = {"status": "Running", "frame": idx + 1, "total_frames": total_frames}
                yield f"data: {json.dumps(progress_data)}\n\n"

            if job_id:
                update_job_info_db(job_id, status="Done")
                create_detection_results_db(job_id, compilance_res)

            final_data = {"status": "Done", "frame": total_frames, "total_frames": total_frames}
            yield f"data: {json.dumps(final_data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return total_frames, 1, 0, inference_generator()
