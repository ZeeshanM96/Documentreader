import cv2
import face_recognition
from google.cloud import vision
import os
from google.cloud.vision_v1 import Image


def initialize_google_cloud():
    """
    Initialize the Google Cloud Vision API credentials.
    """
    credentials_path = "C:\\Users\\zeesh\\Desktop\\Python\\Projects\\FaceTracker\\quiet-odyssey-394722-ba2fca0d219c.json"
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path


def load_known_faces(directory='known_faces'):
    """
    Load known face images, compute encodings, and return them.

    Args:
    - directory (str): Path to directory containing known face images.

    Returns:
    - Tuple[List, List]: Known face encodings and corresponding names.
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found.")
    
    for filepath in os.listdir(directory):
        image_path = os.path.join(directory, filepath)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(filepath.split('-')[0])
    
    return known_face_encodings, known_face_names


def main():
    """
    Main function to capture video, detect and recognize faces.
    """
    initialize_google_cloud()
    
    known_face_encodings, known_face_names = load_known_faces()

    client = vision.ImageAnnotatorClient()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Failed to open the camera.")
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, encoded_image = cv2.imencode('.png', image)

        if not success:
            continue
        
        content = encoded_image.tobytes()
        vision_image = Image(content=content)

        response = client.face_detection(image=vision_image)
        faces = response.face_annotations

        for face in faces:
            vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
            cv2.rectangle(image, vertices[0], vertices[2], (0, 255, 0), 2)

            face_location = (vertices[0][1], vertices[2][0], vertices[2][1], vertices[0][0])
            face_encoding = face_recognition.face_encodings(image, [face_location])[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
            
            cv2.putText(image, name, vertices[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
