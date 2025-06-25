"""
Open cam, take picture, show it, save it
"""
import cv2
import time

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
   print("Error: Could not open camera")
   exit()

print("Warming up camera...")

# Let the camera warm up by capturing and discarding a few frames
for i in range(30):
   ret, frame = cap.read()
   time.sleep(0.1)  # Small delay between frames

print("Camera ready!")

# Capture the actual frame we want to keep
ret, frame = cap.read()

if ret:
   # Display the captured image
   cv2.imshow('Captured Image', frame)
   print("Press any key to save the image...")
   
   # Wait for a key press
   cv2.waitKey(0)
   
   # Save the image to current directory
   cv2.imwrite('captured_image.jpg', frame)
   print("Image saved as 'captured_image.jpg'")
   
   # Close the display window
   cv2.destroyAllWindows()
else:
   print("Error: Could not capture image")

# Release the camera
cap.release()