import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, LargeBinary, Float
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.database import Base



# ======================
# USERS
# ======================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RELATIONSHIPS
    settings = relationship("Setting", back_populates="owner", uselist=False, cascade="all, delete")
    jobs = relationship("DetectionJobs", back_populates="owner", cascade="all, delete")


# ======================
# SETTINGS (1:1)
# ======================
class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)

    top_roi = Column(Integer, default=25)
    bottom_roi = Column(Integer, default=75)
    entry_direction = Column(String, default="top")
    frame_interval = Column(Integer, default=2)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RELATIONSHIP
    owner = relationship("User", back_populates="settings")


# ======================
# DETECTION JOBS
# ======================
class DetectionJobs(Base):
    __tablename__ = "detection_jobs"

    id = Column(String, primary_key=True, index=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # metadata
    name = Column(String)
    video_datetime = Column(DateTime)
    thumbnail_path = Column(String)

    # status
    job_status = Column(String, default="Running")  # Running, Done, Failed, Error
    stored_status = Column(String, default="Not Decided")  # Not Decided, Stored, Not Stored

    # video info
    total_frames = Column(Integer, default=0)
    video_fps = Column(Float)
    video_duration = Column(Float)  # in seconds
    video_path = Column(String)
    video_result_path = Column(String)

    video_datetime_end = Column(DateTime)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RELATIONSHIPS
    owner = relationship("User", back_populates="jobs")
    detections = relationship(
        "DetectionResult",
        back_populates="job",
        cascade="all, delete"
    )


# ======================
# DETECTION RESULTS (per person)
# ======================
class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("detection_jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    # PPE detection (current design)
    apron = Column(Boolean, default=False)
    gloves = Column(Boolean, default=False)
    boots = Column(Boolean, default=False)
    mask = Column(Boolean, default=False)
    hairnet = Column(Boolean, default=False)

    person_track_id = Column(Integer)

    image_data = Column(LargeBinary)
    detection_time = Column(DateTime)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RELATIONSHIP
    job = relationship("DetectionJobs", back_populates="detections")