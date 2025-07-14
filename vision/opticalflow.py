import cv2
import numpy as np

def main():
    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Colors for drawing tracks
    colors = np.random.randint(0, 255, (100, 3))
    
    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect initial corners to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create a mask image for drawing tracks
    mask = np.zeros_like(old_frame)
    
    print("Starting optical flow demo...")
    print("Press 'r' to reset tracking points")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow if we have points to track
        if p0 is not None and len(p0) > 0:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )
            
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    
                    # Draw line showing motion
                    mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                    
                    # Draw point
                    frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
                
                # Combine frame with motion trails
                img = cv2.add(frame, mask)
                
                # Update previous frame and points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                img = frame
        else:
            img = frame
        
        # Add text overlay
        cv2.putText(img, f"Tracking {len(p0) if p0 is not None else 0} points", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Press 'r' to reset, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Optical Flow - Motion Tracking', img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking: detect new corners
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
            print("Tracking points reset")
        elif len(p0) < 10:  # Auto-reset if too few points
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if p0 is not None:
                print(f"Auto-reset: detected {len(p0)} new points")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()