import cv2
import face_recognition

def detect_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if face_locations:
        return "Face detected!"
    return "No face detected."
