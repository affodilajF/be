from datetime import datetime, timezone, timedelta
from database.db_models import DetectionJobs, Setting, DetectionResult
from database.database import SessionLocal

WIB = timezone(timedelta(hours=7))

def get_detection_settings_db(user_id: int):
    with SessionLocal() as db:
        try:
            settings = db.query(Setting).filter(Setting.user_id == user_id).first()
            if settings:
                return {
                    "top_roi": settings.top_roi,
                    "bottom_roi": settings.bottom_roi,
                    "entry_direction": settings.entry_direction,
                    "frame_interval": settings.frame_interval
                }
            return {"top_roi": 25, "bottom_roi": 75, "entry_direction": "top", "frame_interval": 2}
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

def create_detection_results_db(job_id: str, compilance_res: dict):
    with SessionLocal() as db:
        try:
            # compilance_res format: {person_id: {apron: bool, gloves: bool, ..., image_path: str}}
            for person_id, items in compilance_res.items():
                new_res = DetectionResult(
                    job_id = job_id,
                    person_track_id = int(person_id) if person_id is not None else None,
                    apron = items.get("apron", False),
                    gloves = items.get("gloves", False),
                    boots = items.get("boots", False),
                    mask = items.get("mask", False),
                    hairnet = items.get("hairnet", False),
                    image_data = items.get("image_data"),
                    detection_time = items.get("detection_time"),
                    created_at = datetime.now(WIB).replace(microsecond=0)
                )
                db.add(new_res)
            db.commit()
        except:
            db.rollback()
            raise
