from sqlalchemy.orm import Session

from database.db_schemas import DetectionParameter
from database.db_models import Setting

def set_parameters(payload: DetectionParameter, user_id: str, db: Session):
    u_id = int(user_id)

    db_setting = db.query(Setting).filter(Setting.user_id == u_id).first()

    if not db_setting:
        db_setting = Setting(user_id=user_id)
        db.add(db_setting)

    # Update (baik data baru atau lama)
    db_setting.top_roi = payload.top_roi
    db_setting.bottom_roi = payload.bottom_roi
    db_setting.entry_direction = payload.entry_direction
    db_setting.frame_interval = payload.frame_interval

    db.commit()
    db.refresh(db_setting)

    return {
        "success": True,
        "data": {
            "top_roi": db_setting.top_roi,
            "bottom_roi": db_setting.bottom_roi,
            "entry_direction": db_setting.entry_direction,
            "frame_interval" : db_setting.frame_interval
        }
    }

def get_parameters(user_id: str, db: Session):
    db_setting = db.query(Setting).filter(Setting.user_id == int(user_id)).first()
    if not db_setting:
        # Default settings
        return {
            "success": True,
            "data": {
                "top_roi": 25,
                "bottom_roi": 75,
                "entry_direction": "top",
                "frame_interval" : 2
            }
        }
    
    return {
        "success": True,
        "data": {
            "top_roi": db_setting.top_roi,
            "bottom_roi": db_setting.bottom_roi,
            "entry_direction": db_setting.entry_direction,
            "frame_interval" : db_setting.frame_interval
        }
    }