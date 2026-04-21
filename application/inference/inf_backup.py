from ultralytics import YOLO
import cv2, os
from datetime import datetime
import numpy as np

from application.inference.load_detection import load_detection_settings


# Load settings sekali saja dan simpan di CONST
DETECTION_SETTINGS = load_detection_settings()

# Contoh akses
print(DETECTION_SETTINGS)
# DETECTION_SETTINGS["top_roi"], DETECTION_SETTINGS["entry_direction"], dst

MODEL_PATH = r"application/models/best.pt"
OUTPUT_DIR = r"application/video_res"
ROI_TOP_PERCENT = DETECTION_SETTINGS["top_roi"]
ROI_BOTTOM_PERCENT = DETECTION_SETTINGS["bottom_roi"]
ENTRY_DIRECTION = DETECTION_SETTINGS["entry_direction"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
model.verbose = False

person_states = {}
compilance_res = {}

def calculate_roi_from_video(cap):
    """
    Menghitung ROI TOP dan BOTTOM berdasarkan ukuran video dan persen ROI.
    """
    global ROI_TOP, ROI_BOTTOM

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ROI_TOP = int(height * (ROI_TOP_PERCENT / 100))
    ROI_BOTTOM = int(height * (ROI_BOTTOM_PERCENT / 100))

    print(f"Video resolution : {width}x{height}")
    print(f"ROI_TOP (pixel)  : {ROI_TOP}")
    print(f"ROI_BOTTOM(pixel): {ROI_BOTTOM}")

def create_video_writer(cap, output_path):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

def draw_roi_lines(frame):
    h,w = frame.shape[:2]
    color=(0,0,255)
    cv2.line(frame,(0,ROI_TOP),(w,ROI_TOP),color,2)
    cv2.line(frame,(0,ROI_BOTTOM),(w,ROI_BOTTOM),color,2)
    cv2.putText(frame,"ROI TOP",(10,ROI_TOP-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    cv2.putText(frame,"ROI BOTTOM",(10,ROI_BOTTOM-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    return frame

def check_apd(person_id, person_box, frame_boxes, frame_classes, overlap_thresh=0.5):
    global compilance_res
    if person_id is None: return
    person_id = int(person_id)
    if person_id not in compilance_res:
        compilance_res[person_id] = {"apron":False,"gloves":False,"boots":False,"mask":False,"hairnet":False}

    x1,y1,x2,y2 = map(int,person_box)
    for i, box in enumerate(frame_boxes):
        cls = int(frame_classes[i])
        label = str(model.names[cls])
        if label=="person": continue

        x1o,y1o,x2o,y2o = map(int,box)
        xA = max(x1,x1o); yA=max(y1,y1o)
        xB = min(x2,x2o); yB=min(y2,y2o)
        inter_area = max(0,xB-xA)*max(0,yB-yA)
        obj_area = max(1,(x2o-x1o)*(y2o-y1o))
        if inter_area/obj_area>=overlap_thresh:
            compilance_res[person_id][label]=True

def check_person_in_roi(boxes, classes, ids, do_apd_check=True):
    global person_states
    status=[]
    if boxes is None or len(boxes)==0: return status

    for i, box in enumerate(boxes):
        cls=int(classes[i]); label=str(model.names[cls])
        if label!="person": continue
        person_id=int(ids[i]) if ids[i] is not None else None
        x1,y1,x2,y2=map(int,box)

        if person_id not in person_states: person_states[person_id]={"bottom_touched":False}
        state=person_states[person_id]

        if y1 <= ROI_BOTTOM <= y2:
            state["bottom_touched"]=True
            roi_text="no"
        elif not state["bottom_touched"] and y1 <= ROI_TOP <= y2:
            roi_text="yes"
            if do_apd_check: check_apd(person_id,box,boxes,classes)
        else:
            roi_text="no"
        status.append((person_id,roi_text))
    return status

def overlay_info(frame, person_status):
    start_y=30; line_height=25
    for idx,(pid,roi_text) in enumerate(person_status):
        txt=f"Person {pid}: {roi_text}" if pid is not None else f"Person: {roi_text}"
        color=(0,255,0) if roi_text=="yes" else (0,0,255)
        y=start_y+idx*line_height
        cv2.putText(frame,txt,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    return frame

def process_frame(frame, do_apd_check=True):
    results=model.track(frame,persist=True,tracker="trackers/bytetrack.yaml",verbose=False)
    r=results[0]

    annotated=r.plot(line_width=2,conf=False,font_size=0.8)
    annotated=draw_roi_lines(annotated)

    boxes=r.boxes.xyxy.cpu().numpy() if r.boxes else np.array([])
    classes=r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.array([])
    ids=r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.array([None]*len(boxes))

    status=check_person_in_roi(boxes,classes,ids,do_apd_check)
    annotated=overlay_info(annotated,status)

    return annotated

# def run_inference(video_path, save_video=True, apd_every_n_frames=5):
#     global person_states, compilance_res
#     person_states={}; compilance_res={}

#     cap=cv2.VideoCapture(video_path)
#     calculate_roi_from_video(cap)
#     fps=cap.get(cv2.CAP_PROP_FPS) or 30

#     output_path=os.path.join(OUTPUT_DIR,f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4") if save_video else None
#     writer=create_video_writer(cap,output_path) if save_video else None

#     frame_count=0
#     while True:
#         ret,frame=cap.read()
#         if not ret: break
#         do_apd_check=(frame_count%apd_every_n_frames==0)
#         annotated=process_frame(frame,do_apd_check)
#         if save_video and writer: writer.write(annotated)
#         frame_count+=1

#     cap.release()
#     if save_video and writer: writer.release()

#     return {"frames_processed":frame_count,"output_video":output_path,"compilance_res":compilance_res}

import os
import cv2
import json
from datetime import datetime

OUTPUT_DIR = "application/output"
RES_FILE = os.path.join("application", "tmp_data", "detection_res.txt")

person_states = {}
compilance_res = {}

def run_inference(video_path, save_video=True, apd_every_n_frames=5):
    global person_states, compilance_res
    person_states = {}
    compilance_res = {}

    # Buka video
    cap = cv2.VideoCapture(video_path)
    calculate_roi_from_video(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Video output
    output_path = (
        os.path.join(OUTPUT_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        if save_video else None
    )
    writer = create_video_writer(cap, output_path) if save_video else None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        do_apd_check = (frame_count % apd_every_n_frames == 0)
        annotated = process_frame(frame, do_apd_check)

        if save_video and writer:
            writer.write(annotated)

        frame_count += 1

    cap.release()
    if save_video and writer:
        writer.release()

    # Simpan hasil ke file detection_res.txt
    os.makedirs(os.path.dirname(RES_FILE), exist_ok=True)
    result_data = {
        "video": os.path.basename(video_path),
        "frames_processed": frame_count,
        "output_video": output_path,
        "compilance_res": compilance_res,
        "timestamp": datetime.now().isoformat()
    }

    with open(RES_FILE, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Tidak ada return
