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
   time.sleep(0.1)

print("Camera ready!")

# Get camera properties for video recording
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # Default to 20 if can't get FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('captured_video.mp4', fourcc, fps, (width, height))

print("Recording 5 seconds of video...")
start_time = time.time()

# Record for 5 seconds
while time.time() - start_time < 5.0:
   ret, frame = cap.read()
   if ret:
       # Write frame to video file
       out.write(frame)
       
       # Display the frame (optional - shows live recording)
       cv2.imshow('Recording...', frame)
       
       # Break if 'q' is pressed (optional early exit)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   else:
       print("Error: Could not capture frame")
       break

print("Recording complete! Video saved as 'captured_video.mp4'")

# Release everything
out.release()
cap.release()
cv2.destroyAllWindows()