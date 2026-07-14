import os
import uuid
import json
import math
from datetime import datetime, timezone, timedelta
import base64
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload

# from inference.inference import run_inference, run_inference_images
from inference.inference_video import run_inference
from inference.inference_images import run_inference_images
from inference.jobs import inference_jobs
from inference.tasks import process_video_task
from database.db_models import DetectionJobs, Setting, DetectionResult
from database.database import SessionLocal

TMP_DIR = "tmp"
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
    print("handle inference upload for video")
    timestamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    os.makedirs(TMP_DIR, exist_ok=True)
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
        image_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=WIB)
    except:
        image_dt = datetime.now(WIB).replace(microsecond=0)

    metadata = {
        "job_id": job_id,
        "name": name,
        "date": date,
        "time": time,
        "user_id": user_id,
        "data_datetime": image_dt.isoformat(),
        "created_at": datetime.now(WIB).replace(microsecond=0).isoformat()
    }

    # Panggil run_inference_images
    total_frames, gen = run_inference_images(tmp_paths, metadata=metadata)
    
    image_dt_end = image_dt # For image, end is same as start

    # Create new detection job in database
    new_detection = DetectionJobs(
        id=job_id,
        user_id=int(user_id),
        name=name,
        data_datetime=image_dt,
        source_type="IMAGE",
        job_status="Running",
        stored_status="Not Decided", 
        total_frames = total_frames,
        # video_fps = None,
        # video_duration = None,
        data_datetime_end = image_dt_end,
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





def get_compliance_stats_service(user_id: str, search: str, start_date: str, end_date: str, db: Session, lang: str = "id"):
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
        # "perJobStats": {}
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

        #  Mengecek apakah ada minimal satu APD yang tidak dipakai pada satu deteksi. Kalau ada, berarti terjadi pelanggaran (violation).
        has_violation = any(not p["val"] for p in ppe_status)
        if has_violation:
            stats["totalViolations"] += 1

        # Use data_datetime from job or detection_time from res
        dt = job.data_datetime
        if dt:
            # Kalo data belum ada timezone maka tempelkan WIB
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=WIB)
            
            date_key = dt.strftime("%Y-%m-%d")
            if date_key not in trend_map:
                trend_map[date_key] = {
                    "total": 0, "violations": 0, "apronFail": 0, "glovesFail": 0,
                    "bootsFail": 0, "maskFail": 0, "hairnetFail": 0
                }
            
            # trend harian
            trend_map[date_key]["total"] += 1 # berapa banyak deteksi di tanggal itu
            if has_violation:
                trend_map[date_key]["violations"] += 1 # berapa banyak pelanggaran di tanggal itu
            
            # trend per jam
            hour = dt.hour
            stats["hourlyTrend"][hour]["total"] += 1 # mengakses slot jam yang sesuai
            if has_violation:
                stats["hourlyTrend"][hour]["violations"] += 1 # berapa banyak pelanggaran di jam itu
        
        for p in ppe_status:
            is_pass = p["val"]
            if is_pass:
                stats[p["countKey"]]["pass"] += 1 # apd yg dipake 
            else:
                stats[p["countKey"]]["fail"] += 1 # apd yg gadipake
                stats["ppeFailCounts"][p["key"]] = stats["ppeFailCounts"].get(p["key"], 0) + 1 # berapa kali tiap apd gagal
                
                # menghitung kegagalan apd
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

        # (keknya ga dipake ini per job stats, di kode fe juga)
        # statistik per job (batch deteksi)
        # berapa total deteksi dan berapa yang melanggar, dikelompokkan berdasarkan job.name
        # if job.name:
        #     if job.name not in stats["perJobStats"]:
        #         stats["perJobStats"][job.name] = {"total": 0, "violations": 0}
        #     stats["perJobStats"][job.name]["total"] += 1
        #     if has_violation:
        #         stats["perJobStats"][job.name]["violations"] += 1

    stats["totalCompliant"] = stats["totalDetections"] - stats["totalViolations"]

    # Calculate final rates for hourly trend
    # di loop per jam
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
    # di looping per hari
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
    verbal_summary = generate_verbal_summary_python(stats, search, start_date, end_date, lang)

    # Separate trends from stats
    # pop -> hapus dari stats tapi masukkin ke trends
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

def generate_verbal_summary_python(stats, search, start_date, end_date, lang="id"):
    # Normalisasi: apa pun selain "en" dianggap "id"
    is_en = (lang == "en")

    if stats["totalDetections"] == 0:
        return "No data found for the selected filters." if is_en else "Tidak ada data untuk filter yang dipilih."

    # Penghubung daftar ("A, B and C" / "A, B dan C")
    conj = " and " if is_en else " dan "

    def join_names(items):
        if len(items) > 1:
            return ", ".join(items[:-1]) + conj + items[-1]
        return items[0]

    # Nama tampilan kelas APD per bahasa
    PPE_LABELS_ID = {
        'Apron': 'Apron', 'Gloves': 'Sarung Tangan', 'Boots': 'Sepatu Boot',
        'Mask': 'Masker', 'Hairnet': 'Hairnet',
    }
    def ppe_label(name):
        return name if is_en else PPE_LABELS_ID.get(name, name)

    # 1. Date Range Detection
    sorted_days = sorted(stats["dailyTrend"], key=lambda x: x["date"])
    min_date = sorted_days[0]["date"] if sorted_days else None

    # Get Today in WIB
    today_wib = datetime.now(WIB).strftime("%Y-%m-%d")

    MONTHS_ID = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
        7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
    }

    def format_date_str(d_str):
        if not d_str: return "N/A"
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            if is_en:
                return dt.strftime("%B %d, %Y")
            return f"{dt.day} {MONTHS_ID[dt.month]} {dt.year}"
        except:
            return d_str

    effective_end_date = end_date or today_wib

    start_fmt = end_fmt = None
    if min_date:
        start_fmt = format_date_str(start_date or min_date)
        end_fmt = format_date_str(effective_end_date)

    # 2. PPE Violations Analysis
    ppe_fail_list = [
        {"name": 'Apron', "fail": stats["apron"]["fail"]},
        {"name": 'Gloves', "fail": stats["gloves"]["fail"]},
        {"name": 'Boots', "fail": stats["boots"]["fail"]},
        {"name": 'Mask', "fail": stats["mask"]["fail"]},
        {"name": 'Hairnet', "fail": stats["hairnet"]["fail"]},
    ]

    # Rekomendasi ditulis sebagai ajakan aksi langsung, bahasa sederhana.
    VIOLATION_RECOMMENDATIONS_EN = {
        'Gloves': "==provide enough gloves at the entrance==",
        'Mask': "==remind workers to wear their mask before entering==",
        'Hairnet': "==check the hairnet stock at the changing area==",
        'Boots': "==check boot availability and sizing==",
        'Apron': "==make sure aprons are always sufficiently stocked==",
    }
    VIOLATION_RECOMMENDATIONS_ID = {
        'Gloves': "==sediakan sarung tangan yang cukup di pintu masuk==",
        'Mask': "==ingatkan pekerja memakai masker sebelum masuk==",
        'Hairnet': "==cek ketersediaan hairnet di area ganti==",
        'Boots': "==cek ketersediaan dan ukuran sepatu boot==",
        'Apron': "==pastikan ketersediaan apron selalu mencukupi==",
    }
    rec_map = VIOLATION_RECOMMENDATIONS_EN if is_en else VIOLATION_RECOMMENDATIONS_ID

    # cari pelanggaran terbanyak dan tersedikit
    max_fail = max(p["fail"] for p in ppe_fail_list)
    min_fail = min(p["fail"] for p in ppe_fail_list)

    # apd dengan pelanggaran terbanyak dari ppe_fail_list
    # misal => most_frequent_violations[0] = {"name": "Gloves", "fail": 12}
    most_frequent_violations = [p for p in ppe_fail_list if p["fail"] == max_fail and p["fail"] > 0]
    # ubah jadi punya style bold (misalnya : **hairnet**)
    most_frequent_names = [f"**{ppe_label(p['name'])}**" for p in most_frequent_violations]
    # cari apd paling patuh (nol pelanggaran)
    highest_compliance_items = [f"**{ppe_label(p['name'])}**" for p in ppe_fail_list if p["fail"] == min_fail and p["fail"] == 0]

    compliant_rate = round((stats["totalCompliant"] / stats["totalDetections"]) * 100) if stats["totalDetections"] > 0 else 0
    violation_rate = round((stats["totalViolations"] / stats["totalDetections"]) * 100) if stats["totalDetections"] > 0 else 0

    total = stats["totalDetections"]
    compliant = stats["totalCompliant"]
    violations = stats["totalViolations"]

    # Dirangkai sebagai narasi mengalir, bukan poin-poin berjudul.
    paragraphs = []

    # Paragraf 1: gambaran umum, APD paling bermasalah, dan yang paling patuh
    if is_en:
        if start_fmt:
            p1 = f"Across **{start_fmt} to {end_fmt}**, the system recorded **{total} workers** entering the production area."
        else:
            p1 = f"So far the system has recorded **{total} workers** entering the production area."
        p1 += (f" Of all of them, **{compliant} ({compliant_rate}%)** were seen fully equipped, "
               f"while **{violations} ({violation_rate}%)** had yet to complete at least one item. ")
    else:
        if start_fmt:
            p1 = f"Sepanjang periode **{start_fmt} hingga {end_fmt}**, sistem merekam **{total} pekerja** yang memasuki ruangan produksi."
        else:
            p1 = f"Sejauh ini sistem telah merekam **{total} pekerja** yang memasuki ruangan produksi."
        p1 += (f" Dari keseluruhan itu, **{compliant} pekerja ({compliant_rate}%)** terpantau sudah mengenakan APD secara lengkap, "
               f"sementara **{violations} pekerja ({violation_rate}%)** tercatat belum melengkapi setidaknya satu perlengkapan. ")

    if most_frequent_names:
        name_list = join_names(most_frequent_names)
        if len(most_frequent_violations) == 1:
            default_rec = "==review its availability and use==" if is_en else "==cek ketersediaan dan pemakaiannya=="
            recommendation = rec_map.get(most_frequent_violations[0]["name"], default_rec)
        else:
            recommendation = "==review the availability and use of these items==" if is_en else "==cek ketersediaan dan pemakaian APD tersebut=="

        if is_en:
            p1 += (f"The main concern was {name_list}, left off **{max_fail} times**, "
                   f"so it would be wise to {recommendation}. ")
        else:
            p1 += (f"Perhatian utama tertuju pada {name_list}, yang tercatat tidak dikenakan sebanyak **{max_fail} kali**, "
                   f"sehingga sebaiknya {recommendation}. ")

    if highest_compliance_items:
        h_list = join_names(highest_compliance_items)
        if is_en:
            p1 += f"On the other hand, {h_list} stayed the most consistent, with no violations at all."
        else:
            p1 += f"Sebaliknya, {h_list} menjadi yang paling konsisten tanpa satu pun pelanggaran."

    # .strip() menghapus spasi/whitespace di awal dan akhir string (ga menyentuh bagian tengah)
    paragraphs.append(p1.strip())

    # Paragraf 2: pola dari hari ke hari dan dari jam ke jam
    p2 = ""
    if sorted_days:
        best_day = max(sorted_days, key=lambda x: x["rate"])
        worst_day = min(sorted_days, key=lambda x: x["rate"])
        if is_en:
            p2 += (f"Looked at day by day, compliance was highest on {format_date_str(best_day['date'])} "
                   f"at **{best_day['rate']}%**, and lowest on {format_date_str(worst_day['date'])} "
                   f"at just **{worst_day['rate']}%**. ")
        else:
            p2 += (f"Bila ditarik dari hari ke hari, kepatuhan tertinggi terjadi pada {format_date_str(best_day['date'])} "
                   f"yang menyentuh **{best_day['rate']}%**, sedangkan yang terendah ada pada {format_date_str(worst_day['date'])} "
                   f"dengan hanya **{worst_day['rate']}%**. ")

    # menyusun narasi tentang jam2 yg tingkat kepatuhannya jatuh ke 0%, kalo ada ya perlu oengawasan lebih
    active_hours = [h for h in stats["hourlyTrend"] if h["total"] > 0]
    if active_hours:
        zero_hours = [f"**{str(h['hour']).zfill(2)}:00**" for h in active_hours if h["rate"] == 0]
        if zero_hours:
            z_list = join_names(zero_hours)
            if is_en:
                p2 += (f"Going hour by hour, it even slipped all the way to **0%** around {z_list}. "
                       f"==These hours should get closer supervision==.")
            else:
                p2 += (f"Menelusuri lebih jauh ke tiap jam, kepatuhan bahkan sempat jatuh ke **0%** pada pukul {z_list}. "
                       f"==Jam-jam tersebut sebaiknya mendapat pengawasan lebih ketat==.")
        else:
            if is_en:
                p2 += "Hour by hour, meanwhile, none ever fell to 0%."
            else:
                p2 += "Adapun bila dilihat per jam, tak ada satu pun rentang waktu yang jatuh hingga 0%."

    if p2.strip():
        paragraphs.append(p2.strip())

    return "\n\n".join(paragraphs)
