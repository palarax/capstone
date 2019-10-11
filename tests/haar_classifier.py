import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    'configuration/haarcascade_frontalface_default.xml')
# Read the input image
vid = cv2.VideoCapture(0)
while True:
    return_value, frame = vid.read()

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    cv2.imshow("SSD results", frame)
