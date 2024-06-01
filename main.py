from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import base64
import cv2
import dlib
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
# import time

app = Flask(__name__)

def decode_base64_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        # print("Error decoding base64 image:", e)
        return None, f"Error decoding base64 image: {str(e)}"

def detect_faces_and_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_detector = dlib.get_frontal_face_detector()
    face_locations = face_detector(rgb_image)
    face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in face_locations]
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings

def collect_face_encodings(images_base64):
    face_encodings = []
    for image_base64 in images_base64:
        image = decode_base64_to_image(image_base64)
        if image is None:
            print("Error decoding image.")
            return None
        face_locations, face_encodings = detect_faces_and_encodings(image)
        if face_locations is None or len(face_locations) == 0:
            print("No face detected in the provided image")
            return None
        elif len(face_locations) > 1:
            print("Multiple faces detected in the provided image")
            return None
        face_encodings.append(face_encodings[0])
    return face_encodings

def match_faces(old_face_encodings, new_face_encodings, tolerance=0.5):
    matches = []
    for new_face_encoding in new_face_encodings:
        match = False
        for old_face_encoding in old_face_encodings:
            face_distance = face_recognition.face_distance([old_face_encoding], new_face_encoding)
            # print("face_distance===================================",face_distance)
            if face_distance <= tolerance:
                match = True
                break
        matches.append(match)
    return any(matches)

# num_threads = 5  # You can adjust this value as needed
# executor = ThreadPoolExecutor(max_workers=num_threads)

def process_request(item):
    old_images_base64 = item.get('old_images_base64', [])
    new_images_base64 = item.get('new_images_base64', [])

    old_face_encodings = collect_face_encodings(old_images_base64)
    new_face_encodings = collect_face_encodings(new_images_base64)

    if old_face_encodings is None or new_face_encodings is None:
        return {'error': 'No face detected in one of the image sets or multiple faces detected.'}, 0

    # start_time = time.time()

    match_result = match_faces(old_face_encodings, new_face_encodings)
    # print("Match result:", match_result)

    # end_time = time.time()

    # process_time = end_time - start_time
    # print("process_time==================================================================",process_time)

    return match_result
    # return {'match_result': match_result}

num_threads = multiprocessing.cpu_count() * 5  # Use twice the number of CPU cores
# print(num_threads)
executor = ThreadPoolExecutor(max_workers=num_threads)
# print(executor)

@app.route('/match_faces', methods=['POST'])
def match_faces_api():
    data = request.json  # Get the list of JSON objects
    
    # match_result = process_request(data)
    
    # return jsonify({'match_result': match_result})

    futures = [executor.submit(process_request, item) for item in data]
    results = [future.result() for future in futures]
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True,threaded=True,host='192.168.1.64')

