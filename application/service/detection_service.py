import os
import uuid
import json
import math
from datetime import datetime, timezone, timedelta
import base64
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload

from inference.inference import run_inference
from inference.jobs import inference_jobs
from inference.tasks import process_video_task
from database.db_models import DetectionJobs, Setting, DetectionResult
from database.database import SessionLocal

TMP_DIR = "tmp"
THUMBNAIL_DIR = "tmp_thumbnail"
WIB = timezone(timedelta(hours=7))

async def handle_inference_upload(
    background_tasks,
    name: str,
    date: str,
    time: str,
    video,
    # thumbnail,
    save_video: bool,
    user_id: str,
    db: Session
):
    print("handle inference upload")
    timestamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())

    tmp_path = os.path.join(TMP_DIR, f"uploaded_{timestamp}_{video.filename}")
    # simpan video sementara
    content = await video.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    # Combine date and time
    try:
        # Assuming format YYYY-MM-DD HH:MM:SS from client
        video_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=WIB)
    except:
        video_dt = datetime.now(WIB).replace(microsecond=0)

    metadata = {
        "job_id": job_id,
        "name": name,
        "date": date,
        "time": time,
        "user_id": user_id,
        "video_datetime": video_dt.isoformat(),
        "created_at": datetime.now(WIB).replace(microsecond=0).isoformat()
    }

    # Panggil run_inference untuk dapat total_frames, fps, durasi dan generator-nya
    total_frames, fps, duration, gen = run_inference(tmp_path, save_video=save_video, metadata=metadata)
    
    video_dt_end = video_dt + timedelta(seconds=duration)

    # Create new detection job in database
    new_detection = DetectionJobs(
        id=job_id,
        user_id=int(user_id),
        name=name,
        video_datetime=video_dt,
        video_path=tmp_path,
        job_status="Running",
        stored_status="Not Decided", 
        total_frames = total_frames,
        video_fps = fps,
        video_duration = duration,
        video_datetime_end = video_dt_end,
        created_at = datetime.now(WIB).replace(microsecond=0)
    )
    db.add(new_detection)
    db.commit()
    db.refresh(new_detection)


    # Still using legacy inference_jobs for SSE tracking
    inference_jobs[job_id] = {
        "status": "Running",
        "progress": {},
        "result": None,
        "new_event": False
    }

    background_tasks.add_task(process_video_task, job_id, gen, tmp_path)

    return {
        "success": True,
        "message": "Inference started",
        "data": {
            "job_id": job_id,
            "total_frames": total_frames,
            "video_fps": fps,
            "video_duration": duration,
            "video_datetime_end": video_dt_end
        }
    }


def get_detection_result_list(user_id: str, db: Session):
    # Get all jobs for the user that are stored, including their detections
    jobs = db.query(DetectionJobs).options(joinedload(DetectionJobs.detections)).filter(
        DetectionJobs.user_id == int(user_id),
        DetectionJobs.stored_status == "Stored"
    ).order_by(DetectionJobs.created_at.desc()).all()

    total_detection_result = 0
    results = []

    for job in jobs:
        # Format detections for this job
        detection_list = []
        for d in job.detections:
            detection_list.append({
                "id": d.id,
                "apron": d.apron,
                "gloves": d.gloves,
                "boots": d.boots,
                "mask": d.mask,
                "hairnet": d.hairnet,
                "person_track_id": d.person_track_id,
                "image_data": base64.b64encode(d.image_data).decode('utf-8') if d.image_data else None,
                "detection_time": d.detection_time,
                "created_at": d.created_at
            })
            total_detection_result += 1
        
        results.append({
            "job_id": job.id,
            "name": job.name,
            "video_datetime": job.video_datetime,
            "video_datetime_end": job.video_datetime_end,
            "video_fps": job.video_fps,
            "video_duration": job.video_duration,
            "created_at": job.created_at,
            "detection_result": detection_list
        })

    return {
        "success": True,
        "data": {
            "total_detection_result": total_detection_result,
            "detection_jobs": results
        }
    }


def get_detection_list(user_id: str, page: int, limit: int, db: Session):
    # Total count of jobs for pagination
    job_query = db.query(DetectionJobs).filter(
        DetectionJobs.user_id == int(user_id),
        DetectionJobs.stored_status == "Stored"
    )
    total_jobs = job_query.count()

    # Total count of all detection results for the user's stored jobs
    total_detection_result = db.query(DetectionResult).join(DetectionJobs).filter(
        DetectionJobs.user_id == int(user_id),
        DetectionJobs.stored_status == "Stored"
    ).count()

    # Paginated jobs
    start = (page - 1) * limit
    jobs = job_query.options(joinedload(DetectionJobs.detections)).order_by(DetectionJobs.created_at.desc()).offset(start).limit(limit).all()

    # Build the response list
    results = []
    for job in jobs:
        # Format detections for this job
        detection_list = []
        for d in job.detections:
            detection_list.append({
                "id": d.id,
                "apron": d.apron,
                "gloves": d.gloves,
                "boots": d.boots,
                "mask": d.mask,
                "hairnet": d.hairnet,
                "person_track_id": d.person_track_id,
                "image_data": base64.b64encode(d.image_data).decode('utf-8') if d.image_data else None,
                "detection_time": d.detection_time,
                "created_at": d.created_at
            })
        
        results.append({
            "job_id": job.id,
            "job_status": job.job_status,
            "stored_status": job.stored_status,
            "name": job.name,
            "video_datetime": job.video_datetime,
            "video_datetime_end": job.video_datetime_end,
            "video_fps": job.video_fps,
            "video_duration": job.video_duration,
            "total_frames": job.total_frames,
            "created_at": job.created_at,
            "detection_result_items": detection_list  
        })
    
    return {
        "success": True,
        "data": {
            "total_jobs": total_jobs,
            "total_detection_result": total_detection_result,
            # "total": total_jobs,  # Optional: kept for backward compatibility if any
            "total_pages": (total_jobs + limit - 1) // limit,
            "page": page,
            "limit": limit,
            "detection_jobs": results
        }
    }

def get_detection_results_by_job_id(job_id, db: Session):
    detections = db.query(DetectionResult).filter(
        DetectionResult.job_id == job_id
    ).all()

    detection_list = []
    for d in detections:
        detection_list.append({
            "id": d.id,
            "apron": d.apron,
            "gloves": d.gloves,
            "boots": d.boots,
            "mask": d.mask,
            "hairnet": d.hairnet,
            "person_track_id": d.person_track_id,
            "image_data": base64.b64encode(d.image_data).decode('utf-8') if d.image_data else None,
            "detection_time": d.detection_time,
            "created_at": d.created_at
        })

    return {
        "success": True,
        "data": {
            "job_id": job_id,
            "detection_result_items": detection_list  
        }
    }

def get_not_decided_detection(user_id: str, db: Session):
    # Ambil job terbaru
    job = db.query(DetectionJobs).filter(
        DetectionJobs.user_id == int(user_id),
        DetectionJobs.stored_status == "Not Decided"
    ).order_by(DetectionJobs.created_at.desc()).first()
    
    # Jika tidak ada job, return langsung, gausah diganti error
    if not job:
        return {
            "success": True,
            "data": None
        }

    detections = db.query(DetectionResult).filter(
        DetectionResult.job_id == job.id
    ).all()  # .all() agar jadi list

    # Format detections menjadi list dict agar bisa di-JSON encode
    detection_list = []
    for d in detections:
        detection_list.append({
            "id": d.id,
            "apron": d.apron,
            "gloves": d.gloves,
            "boots": d.boots,
            "mask": d.mask,
            "hairnet": d.hairnet,
            "person_track_id": d.person_track_id,
            "image_data": base64.b64encode(d.image_data).decode('utf-8') if d.image_data else None,
            "detection_time": d.detection_time,
            "created_at": d.created_at
        })
    
    return {
        "success": True,
        "data": {
            "job_id": job.id,
            "job_status": job.job_status,
            "stored_status": job.stored_status,
            "name": job.name,
            "video_datetime": job.video_datetime,
            "video_datetime_end": job.video_datetime_end,
            "video_fps": job.video_fps,
            "video_duration": job.video_duration,
            "total_frames": job.total_frames,
            "created_at": job.created_at,
            "detection_result_items": detection_list  
        }
    }


def update_detection_store_status(store_status, job_id: str, db: Session):
    job = db.query(DetectionJobs).filter(DetectionJobs.id == job_id).first()
    if job:
        job.stored_status = store_status
        db.commit()
        return {
            "success": True,
            "message": "Detection result status updated to " + store_status,
            "data": {"job_id": job_id}
        }
    
    return JSONResponse(status_code=404, content={"message": "Job not found"})


def get_detection_settings_db(user_id: int):
    with SessionLocal() as db:
        try:
            settings = db.query(Setting).filter(Setting.user_id == user_id).first()
            if settings:
                return {
                    "top_roi": settings.top_roi,
                    "bottom_roi": settings.bottom_roi,
                }
            return {"top_roi": 25, "bottom_roi": 75}
        except:
            raise

def update_job_info_db(job_id: str, status: str = None, total_frames: int = None, video_result_path: str = None):
    with SessionLocal() as db:
        try:
            job = db.query(DetectionJobs).filter(DetectionJobs.id == job_id).first()
            if job:
                if status: job.job_status = status
                if total_frames is not None: job.total_frames = total_frames
                if video_result_path: job.video_result_path = video_result_path
                db.commit()
        except:
            db.rollback()
            raise


