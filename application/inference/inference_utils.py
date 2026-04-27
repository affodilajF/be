from ultralytics.utils.plotting import Annotator, colors
import cv2
import numpy as np

def is_overlapping(box1, box2, thresh=0.5):
    """Cek apakah box1 (APD) berada di dalam/beririsan dengan box2 (Person)"""
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return inter / max(1, area1) >= thresh

def calculate_roi_from_video(cap, top_percent, bottom_percent):
    """Calculate ROI TOP and BOTTOM based on video dimensions and percentage."""
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_top = int(height * (top_percent / 100))
    roi_bottom = int(height * (bottom_percent / 100))
    return roi_top, roi_bottom

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

def save_person_crop(person_id, person_box, frame, r):
    # 1. Gunakan Annotator pada salinan frame asli
    ann = Annotator(frame.copy(), line_width=2)
    
    if r.boxes is not None:
        for d in r.boxes:
            box = d.xyxy[0]
            cls = int(d.cls)
            label = r.names[cls]
            
            should_draw = False
            if label == "person":
                if d.id is not None:
                    if int(d.id) == person_id:
                        should_draw = True
                else:
                    # For images without track IDs, match by box coordinates
                    if np.allclose(box.cpu().numpy(), person_box, atol=1e-3):
                        should_draw = True
            else:
                if is_overlapping(box, person_box):
                    should_draw = True
            
            if should_draw:
                ann.box_label(box, label, color=colors(cls, True))

    work_frame = ann.result()

    # 2. Proses Crop
    px1, py1, px2, py2 = map(int, person_box)
    pad = 50 
    h, w = work_frame.shape[:2]
    px1, py1, px2, py2 = max(0, px1-pad), max(0, py1-pad), min(w, px2+pad), min(h, py2+pad)
    
    crop = work_frame[py1:py2, px1:px2]
    if crop.size > 0:
        success, buffer = cv2.imencode(".jpg", crop)
        if success:
            return buffer.tobytes()
    return None

def overlay_info(frame, person_status):
    start_y = 30; line_height = 25
    for idx, (pid, roi_text) in enumerate(person_status):
        txt = f"Person {pid}: {roi_text}" if pid is not None else f"Person: {roi_text}"
        color = (0, 255, 0) if roi_text == "yes" else (0, 0, 255)
        y = start_y + idx * line_height
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
