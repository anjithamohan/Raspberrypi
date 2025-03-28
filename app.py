from flask import Flask, request, render_template, jsonify
from sensor_handlers.camera import capture_image
from sensor_handlers.microphone import process_audio
from sensor_handlers.speaker import announce
from sensor_handlers.motion_detector import detect_motion
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='logs/events.log', level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET'])
def capture():
    image_path = capture_image()
    return jsonify({"status": "success", "image": image_path})

@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    data = request.json
    detected_event = process_audio(data['audio'])
    if detected_event:
        announce(detected_event)
    return jsonify({"event": detected_event})

@app.route('/motion_detected', methods=['POST'])
def motion():
    detected = detect_motion()
    if detected:
        announce("Unauthorized movement detected! Security alert triggered.")
    return jsonify({"motion_detected": detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
