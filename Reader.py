import cv2
import numpy as np
from google.cloud import vision
import os


def order_points(pts):
    """
    Orders the given set of four points for perspective transformation.
    
    Args:
        pts (list of tuple): List containing four (x, y) coordinates.
        
    Returns:
        numpy.ndarray: Sorted coordinates in the order top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Performs a four-point perspective transformation on the given image.
    
    Args:
        image (numpy.ndarray): Input image.
        pts (list of tuple): List containing four (x, y) coordinates.
        
    Returns:
        numpy.ndarray: Warped image after perspective transformation.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document_corners(image_path):
    """
    Detect the four corners of the largest contour in the image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: List of four corner points if detected, else None.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def document_scan_and_ocr(image_path, credentials_path):
    """
    Scans the provided document and performs OCR using Google Cloud Vision API.
    
    Args:
        image_path (str): Path to the image file.
        credentials_path (str): Path to the Google Cloud credentials JSON file.
        
    Returns:
        None: Prints the detected text to the console.
    """
    # Set up the credentials for Google Cloud Vision API
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Instantiate a client
    client = vision.ImageAnnotatorClient()

    # Load the image from file
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
    except FileNotFoundError:
        print(f"Error: {image_path} not found.")
        return

    # By default, use the original image for OCR.
    image = vision.Image(content=content)

    corners = detect_document_corners(image_path)
    if corners is not None:
        warped_image = four_point_transform(cv2.imread(image_path), corners)
        # Convert the warped image back to the format required by Vision API.
        _, buf = cv2.imencode(".png", warped_image)
        image = vision.Image(content=buf.tobytes())

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Print the detected text
    for text in texts:
        print(text.description)



if __name__ == "__main__":
    # image and gcp credentials path.
    image_path = "C:\\Users\\zeesh\\Desktop\\Python\\Opencv-project\\Documentreader\\image.png"
    credentials_path = "C:\\Users\\zeesh\\Desktop\\Python\\Opencv-project\\Documentreader\\quiet-odyssey-394722-ba2fca0d219c.json"

    document_scan_and_ocr(image_path, credentials_path)
