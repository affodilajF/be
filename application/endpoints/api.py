from fastapi import APIRouter, UploadFile, File, Form, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from database.database import get_db
from app.auth_bearer import JWTBearer
from app.auth_handler import decodeJWT
import json
import asyncio
from database.db_schemas import DetectionParameter
from service.detection_service import (
    handle_inference_upload,
    handle_inference_upload_images,
    get_detection_list,
    get_not_decided_detection,
    update_detection_store_status,
    get_detection_results_by_job_id,
    get_detection_result_list
)
from service.settings_service import (
    set_parameters,
    get_parameters,
)
from inference.jobs import inference_jobs

# router = APIRouter(dependencies=[Depends(JWTBearer())])
router = APIRouter()

## acc
@router.post("/api/run-ai-model")
async def inference_upload(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    video: UploadFile = File(...),
    # thumbnail: str = File(...),
    save_video: bool = Form(True),
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    payload = decodeJWT(token)
    user_id = payload.get("user_id")
    print(date)
    print(time)
    
    return await handle_inference_upload(
        background_tasks, name, date, time, video, save_video, user_id, db
    )

## acc
@router.post("/api/run-ai-model-images")
async def inference_upload_images(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    images: list[UploadFile] = File(...),
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    payload = decodeJWT(token)
    user_id = payload.get("user_id")
    
    return await handle_inference_upload_images(
        background_tasks, name, date, time, images, user_id, db
    )

# acc
@router.get("/api/inference-status/{job_id}")
async def get_inference_status(job_id: str):
    async def event_generator():
        while True:
            job = inference_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                break
            
            if job["new_event"]:
                job["new_event"] = False
                if job["status"] == "Done":
                    yield f"data: {json.dumps(job['result'])}\n\n"
                    break
                else:
                    yield f"data: {json.dumps(job['progress'])}\n\n"
            elif job["status"] == "Done":
                yield f"data: {json.dumps(job['result'])}\n\n"
                break
                
            await asyncio.sleep(1)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# acc
@router.post("/api/set-detection-parameter")
async def parameter_detection(
    payload: DetectionParameter, 
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    token_payload = decodeJWT(token)
    user_id = token_payload.get("user_id")
    return set_parameters(payload, user_id, db)

# acc
@router.get("/api/get-detection-parameter")
async def get_detection_parameter(
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    token_payload = decodeJWT(token)
    user_id = token_payload.get("user_id")
    return get_parameters(user_id, db)



# acc
@router.get("/api/get-list-detection-result")
async def get_detection_result(
    job_id: str,
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    return get_detection_results_by_job_id(job_id, db)

# acc
@router.get("/api/get-not-decided-detection")
async def get_not_decided_detection_data(
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    payload = decodeJWT(token)
    user_id = payload.get("user_id")
    return get_not_decided_detection(user_id, db)

# acc
@router.get("/api/set-detection-store-status")
async def set_detection_store_status(
    job_id: str,
    store_status: str,
    db: Session = Depends(get_db)
):
    return update_detection_store_status(store_status, job_id, db)

# acc
@router.get("/api/detection-list-data")
async def detection_list_data(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1),
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    payload = decodeJWT(token)
    user_id = payload.get("user_id")
    return get_detection_list(user_id, page, limit, db)

# # for dahsboard stat
@router.get("/api/detection-result-list-data")
async def detection_result_list_data(
    token: str = Depends(JWTBearer()),
    db: Session = Depends(get_db)
):
    payload = decodeJWT(token)
    user_id = payload.get("user_id")
    return get_detection_result_list(user_id, db)
