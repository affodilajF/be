import json
import os
import time
from inference.jobs import inference_jobs
from service.inference_db_service import update_job_info_db

def process_video_task(job_id, gen, tmp_path):
    try:
        for sse_data in gen:
            # Mengambil data dict dari string "data: {...}\n\n"
            data_dict = json.loads(sse_data.replace("data: ", "").strip())
            
            if data_dict.get("status") == "Done":
                inference_jobs[job_id]["status"] = "Done"
                inference_jobs[job_id]["result"] = data_dict
                inference_jobs[job_id]["new_event"] = True
                
                # Update DB to Done
                # update_job_info_db(job_id, status="Done", total_frames=data_dict.get("total_frames"))
                
                # Tunggu agar SSE terkirim ke front-end sebelum status/jika akan dihapus
                time.sleep(3) 
            elif data_dict.get("status") == "Error":
                inference_jobs[job_id]["status"] = "Error"
                inference_jobs[job_id]["result"] = data_dict
                inference_jobs[job_id]["new_event"] = True
                update_job_info_db(job_id, status="Failed")
                time.sleep(3)
            else:
                inference_jobs[job_id]["status"] = "Running"
                inference_jobs[job_id]["progress"] = data_dict
                inference_jobs[job_id]["new_event"] = True

                # Update total_frames di DB jika ada perubahan/info baru
                if "total_frames" in data_dict:
                    update_job_info_db(job_id, total_frames=data_dict["total_frames"])
    finally:
        # Hapus file sementara setelah selesai atau jika error
        if isinstance(tmp_path, list):
            for p in tmp_path:
                if os.path.exists(p):
                    os.remove(p)
        elif tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
