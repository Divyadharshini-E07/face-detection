import cv2
import os

# Define the folder path where your files are located
folder_path = 'C:/Users/Divya/Desktop/FaceDetectionProject/'

# Define the absolute paths to the Haar cascade file and the image file within the folder
haar_cascade_path = os.path.join(folder_path, 'haarcascade_frontalface_default.xml')
image_path = os.path.join(folder_path, 'test.png')
output_path = os.path.join(folder_path, 'face_detected.jpg')

# Check if the Haar cascade file exists
if not os.path.exists(haar_cascade_path):
    print(f"Error: The Haar cascade file was not found at {haar_cascade_path}.")
    exit()

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the Haar cascade was loaded successfully
if face_cascade.empty():
    print(f"Error: Failed to load Haar cascade from {haar_cascade_path}.")
    exit()

# Read the image file
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Error: Could not open or find the image at {image_path}.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save the result
cv2.imwrite(output_path, img)
print(f"Face detected image saved as {output_path}")

# Read and resize the output image
output_image = cv2.imread(output_path)
max_width = 800
max_height = 600
height, width = output_image.shape[:2]
scaling_factor = min(max_width / width, max_height / height)
resized_image = cv2.resize(output_image, (int(width * scaling_factor), int(height * scaling_factor)))

# Display the output image
cv2.imshow('Detected Faces', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
