import os
import uuid
import json
import math
from datetime import datetime, timezone, timedelta
import base64
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload

from inference.inference import run_inference, run_inference_images
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
        "data_datetime": video_dt.isoformat(),
        "created_at": datetime.now(WIB).replace(microsecond=0).isoformat()
    }

    # Panggil run_inference untuk dapat total_frames, fps, durasi dan generator-nya
    total_frames, fps, duration, gen = run_inference(tmp_path, save_video=save_video, metadata=metadata)
    
    video_dt_end = (video_dt + timedelta(seconds=duration)).replace(microsecond=0)

    # Create new detection job in database
    new_detection = DetectionJobs(
        id=job_id,
        user_id=int(user_id),
        name=name,
        data_datetime=video_dt,
        source_type="VIDEO",
        job_status="Running",
        stored_status="Not Decided", 
        total_frames = total_frames,
        video_fps = fps,
        video_duration = duration,
        data_datetime_end = video_dt_end,
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


async def handle_inference_upload_images(
    background_tasks,
    name: str,
    date: str,
    time: str,
    images: list,
    user_id: str,
    db: Session
):
    print(f"handle inference upload images: {len(images)} images")
    timestamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())

    tmp_paths = []
    for idx, image in enumerate(images):
        tmp_path = os.path.join(TMP_DIR, f"uploaded_{timestamp}_{idx}_{image.filename}")
        content = await image.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        tmp_paths.append(tmp_path)

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
        "data_datetime": video_dt.isoformat(),
        "created_at": datetime.now(WIB).replace(microsecond=0).isoformat()
    }

    # Panggil run_inference_images
    total_frames, fps, duration, gen = run_inference_images(tmp_paths, metadata=metadata)
    
    video_dt_end = video_dt # For image, end is same as start

    # Create new detection job in database
    new_detection = DetectionJobs(
        id=job_id,
        user_id=int(user_id),
        name=name,
        data_datetime=video_dt,
        source_type="IMAGE",
        job_status="Running",
        stored_status="Not Decided", 
        total_frames = total_frames,
        video_fps = fps,
        video_duration = duration,
        data_datetime_end = video_dt_end,
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

    background_tasks.add_task(process_video_task, job_id, gen, tmp_paths)

    return {
        "success": True,
        "message": f"Inference started for {len(images)} images",
        "data": {
            "job_id": job_id,
            "total_frames": total_frames,
            # "video_fps": fps,
            # "video_duration": duration,
            # "video_datetime_end": video_dt_end
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
            "source_type": job.source_type,
            "data_datetime": job.data_datetime,
            "data_datetime_end": job.data_datetime_end,
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
            "source_type": job.source_type,
            "stored_status": job.stored_status,
            "name": job.name,
            "data_datetime": job.data_datetime,
            "data_datetime_end": job.data_datetime_end,
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
            "source_type": job.source_type,
            "stored_status": job.stored_status,
            "name": job.name,
            "data_datetime": job.data_datetime,
            "data_datetime_end": job.data_datetime_end,
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





def get_compliance_stats_service(user_id: str, search: str, start_date: str, end_date: str, db: Session):
    # Fetch detections joined with jobs
    query = db.query(DetectionResult, DetectionJobs).join(DetectionJobs).filter(
        DetectionJobs.user_id == int(user_id),
        DetectionJobs.stored_status == "Stored"
    )

    if search:
        query = query.filter(DetectionJobs.name.ilike(f"%{search}%"))
    
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=WIB)
            query = query.filter(DetectionJobs.data_datetime >= start_dt)
        except Exception as e:
            print(f"Error parsing start_date: {e}")

    if end_date:
        try:
            # End of day
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=WIB) + timedelta(days=1) - timedelta(seconds=1)
            query = query.filter(DetectionJobs.data_datetime <= end_dt)
        except Exception as e:
            print(f"Error parsing end_date: {e}")

    results = query.order_by(DetectionJobs.data_datetime.asc()).all()

    stats = {
        "apron": {"pass": 0, "fail": 0},
        "gloves": {"pass": 0, "fail": 0},
        "boots": {"pass": 0, "fail": 0},
        "mask": {"pass": 0, "fail": 0},
        "hairnet": {"pass": 0, "fail": 0},
        "totalDetections": len(results),
        "totalViolations": 0,
        "totalCompliant": 0,
        "complianceScore": 100,
        "dailyTrend": [],
        "hourlyTrend": [{"hour": i, "rate": 100, "apronFail": 0, "glovesFail": 0, "bootsFail": 0, "maskFail": 0, "hairnetFail": 0, "total": 0, "violations": 0} for i in range(24)],
        "ppeFailCounts": {},
        "perJobStats": {}
    }

    trend_map = {}

    for res, job in results:
        ppe_status = [
            {"key": "Apron", "val": res.apron, "countKey": "apron"},
            {"key": "Gloves", "val": res.gloves, "countKey": "gloves"},
            {"key": "Boots", "val": res.boots, "countKey": "boots"},
            {"key": "Mask", "val": res.mask, "countKey": "mask"},
            {"key": "Hairnet", "val": res.hairnet, "countKey": "hairnet"},
        ]

        has_violation = any(not p["val"] for p in ppe_status)
        if has_violation:
            stats["totalViolations"] += 1

        # Use data_datetime from job or detection_time from res
        dt = job.data_datetime
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=WIB)
            
            date_key = dt.strftime("%Y-%m-%d")
            if date_key not in trend_map:
                trend_map[date_key] = {
                    "total": 0, "violations": 0, "apronFail": 0, "glovesFail": 0,
                    "bootsFail": 0, "maskFail": 0, "hairnetFail": 0
                }
            
            trend_map[date_key]["total"] += 1
            if has_violation:
                trend_map[date_key]["violations"] += 1
            
            hour = dt.hour
            stats["hourlyTrend"][hour]["total"] += 1
            if has_violation:
                stats["hourlyTrend"][hour]["violations"] += 1
        
        for p in ppe_status:
            is_pass = p["val"]
            if is_pass:
                stats[p["countKey"]]["pass"] += 1
            else:
                stats[p["countKey"]]["fail"] += 1
                stats["ppeFailCounts"][p["key"]] = stats["ppeFailCounts"].get(p["key"], 0) + 1
                
                if dt:
                    if p["key"] == "Apron":
                        trend_map[date_key]["apronFail"] += 1
                        stats["hourlyTrend"][hour]["apronFail"] += 1
                    elif p["key"] == "Gloves":
                        trend_map[date_key]["glovesFail"] += 1
                        stats["hourlyTrend"][hour]["glovesFail"] += 1
                    elif p["key"] == "Boots":
                        trend_map[date_key]["bootsFail"] += 1
                        stats["hourlyTrend"][hour]["bootsFail"] += 1
                    elif p["key"] == "Mask":
                        trend_map[date_key]["maskFail"] += 1
                        stats["hourlyTrend"][hour]["maskFail"] += 1
                    elif p["key"] == "Hairnet":
                        trend_map[date_key]["hairnetFail"] += 1
                        stats["hourlyTrend"][hour]["hairnetFail"] += 1

        if job.name:
            if job.name not in stats["perJobStats"]:
                stats["perJobStats"][job.name] = {"total": 0, "violations": 0}
            stats["perJobStats"][job.name]["total"] += 1
            if has_violation:
                stats["perJobStats"][job.name]["violations"] += 1

    stats["totalCompliant"] = stats["totalDetections"] - stats["totalViolations"]

    # Calculate final rates for hourly trend
    for h in stats["hourlyTrend"]:
        if h["total"] > 0:
            h["rate"] = round(((h["total"] - h["violations"]) / h["total"]) * 100)
        else:
            h["rate"] = 100

    if stats["totalDetections"] > 0:
        stats["complianceScore"] = round(
            ((stats["totalDetections"] - stats["totalViolations"]) / stats["totalDetections"]) * 100
        )

    # Daily Trend
    sorted_dates = sorted(trend_map.keys())
    for d_key in sorted_dates:
        t = trend_map[d_key]
        stats["dailyTrend"].append({
            "date": d_key,
            "rate": round(((t["total"] - t["violations"]) / t["total"]) * 100) if t["total"] > 0 else 100,
            "apronFail": t["apronFail"],
            "glovesFail": t["glovesFail"],
            "bootsFail": t["bootsFail"],
            "maskFail": t["maskFail"],
            "hairnetFail": t["hairnetFail"],
            "total": t["total"],
            "violations": t["violations"]
        })

    # Generate verbal summary
    verbal_summary = generate_verbal_summary_python(stats, search, start_date, end_date)

    # Separate trends from stats
    trends = {
        "dailyTrend": stats.pop("dailyTrend"),
        "hourlyTrend": stats.pop("hourlyTrend")
    }

    return {
        "success": True,
        "data": {
            "stats": stats,
            "trends": trends,
            "summary": verbal_summary
        }
    }

def generate_verbal_summary_python(stats, search, start_date, end_date):
    if stats["totalDetections"] == 0:
        return "No data found for the selected filters."

    # 1. Date Range Detection
    sorted_days = sorted(stats["dailyTrend"], key=lambda x: x["date"])
    min_date = sorted_days[0]["date"] if sorted_days else None
    max_date = sorted_days[-1]["date"] if sorted_days else None

    # Get Today in WIB
    today_wib = datetime.now(WIB).strftime("%Y-%m-%d")

    def format_date_str(d_str):
        if not d_str: return "N/A"
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        except:
            return d_str

    effective_end_date = end_date or today_wib
    
    observation_period = ""
    if min_date:
        start_fmt = format_date_str(start_date or min_date)
        end_fmt = format_date_str(effective_end_date)
        observation_period = f"over the observation period of **{start_fmt} to {end_fmt}**"

    # 2. PPE Violations Analysis
    ppe_fail_list = [
        {"name": 'Apron', "fail": stats["apron"]["fail"]},
        {"name": 'Gloves', "fail": stats["gloves"]["fail"]},
        {"name": 'Boots', "fail": stats["boots"]["fail"]},
        {"name": 'Mask', "fail": stats["mask"]["fail"]},
        {"name": 'Hairnet', "fail": stats["hairnet"]["fail"]},
    ]

    VIOLATION_RECOMMENDATIONS = {
        'Gloves': "==glove availability at entrance be reviewed==",
        'Mask': "==workers be reminded of mask protocol before entering==",
        'Hairnet': "==hairnet stock at changing area be checked==",
        'Boots': "==boot storage and sizing availability be inspected==",
        'Apron': "==apron supply be ensured sufficient per shift==",
    }

    max_fail = max(p["fail"] for p in ppe_fail_list)
    min_fail = min(p["fail"] for p in ppe_fail_list)

    most_frequent_violations = [p for p in ppe_fail_list if p["fail"] == max_fail and p["fail"] > 0]
    most_frequent_names = [f"**{p['name']}**" for p in most_frequent_violations]
    highest_compliance_items = [f"**{p['name']}**" for p in ppe_fail_list if p["fail"] == min_fail and p["fail"] == 0]

    compliant_rate = round((stats["totalCompliant"] / stats["totalDetections"]) * 100) if stats["totalDetections"] > 0 else 0
    violation_rate = round((stats["totalViolations"] / stats["totalDetections"]) * 100) if stats["totalDetections"] > 0 else 0

    # Paragraph 1: Overview & PPE
    summary = f"Based on all-time recorded data {observation_period}, a total of **{stats['totalDetections']} workers** were observed, with **{stats['totalCompliant']} workers ({compliant_rate}%)** fully compliant and **{stats['totalViolations']} workers ({violation_rate}%)** recorded with at least one PPE violation. "

    if most_frequent_names:
        if len(most_frequent_names) > 1:
            name_list = ", ".join(most_frequent_names[:-1]) + " and " + most_frequent_names[-1]
        else:
            name_list = most_frequent_names[0]

        if len(most_frequent_violations) == 1:
            recommendation = VIOLATION_RECOMMENDATIONS.get(most_frequent_violations[0]["name"], "==availability and protocol adherence be reviewed==")
        else:
            recommendation = "==availability and protocol adherence for these items be reviewed=="

        summary += f"The most frequent violation was the absence of {name_list} ({max_fail} cases), and it is recommended that {recommendation}. "

    if highest_compliance_items:
        if len(highest_compliance_items) > 1:
            h_list = ", ".join(highest_compliance_items[:-1]) + " and " + highest_compliance_items[-1]
        else:
            h_list = highest_compliance_items[0]
        summary += f"{h_list} recorded no violations throughout the observation period.\n\n"
    else:
        summary += "\n\n"

    # Paragraph 2: Daily Trend
    if sorted_days:
        best_day = max(sorted_days, key=lambda x: x["rate"])
        worst_day = min(sorted_days, key=lambda x: x["rate"])

        summary += f"Daily trend analysis shows the best compliance was recorded on {format_date_str(best_day['date'])} (**{best_day['rate']}%**), while the worst was on {format_date_str(worst_day['date'])} (**{worst_day['rate']}%**).\n\n"

    # Paragraph 3: Hourly Trend
    active_hours = [h for h in stats["hourlyTrend"] if h["total"] > 0]
    if active_hours:
        zero_hours = [f"**{str(h['hour']).zfill(2)}:00**" for h in active_hours if h["rate"] == 0]

        summary += "Hourly trend analysis reveals compliance rate was variable throughout the day, "

        if zero_hours:
            if len(zero_hours) > 1:
                z_list = ", ".join(zero_hours[:-1]) + " and " + zero_hours[-1]
            else:
                z_list = zero_hours[0]
            summary += f"with notable drops to **0%** at {z_list}. ==Targeted supervision is recommended during these hours== to improve overall compliance."
        else:
            summary += "with no hours showing complete non-compliance. ==Continued monitoring is recommended== to identify recurring low-compliance patterns."

    return summary
