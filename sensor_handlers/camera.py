import cv2

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    image_path = "static/captured_image.jpg"
    if ret:
        cv2.imwrite(image_path, frame)
    cap.release()
    return image_path
