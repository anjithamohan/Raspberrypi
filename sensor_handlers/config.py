import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "supersecretkey"
    DEBUG = True
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"wav", "mp3", "jpg", "png", "mp4"}
