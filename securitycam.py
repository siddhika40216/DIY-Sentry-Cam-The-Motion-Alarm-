import cv2        # Library for image processing/computer vision
import winsound   # Library to play the Windows system beep

# Connect to the default camera (ID 0)
webcam = cv2.VideoCapture(0)

# Step 1: Capture the very first frame to start the comparison
ret, img1 = webcam.read()

# Main loop: Runs as long as the camera is connected
while webcam.isOpened():
    
    # Step 2: Capture a new frame to compare against the previous one
    ret, img2 = webcam.read()
    if not ret:
        break # Exit if the camera stops sending data

    # Step 3: Find the difference between the two images
    # This creates a 'ghost' image of only what moved
    diff = cv2.absdiff(img1, img2)
    
    # Step 4: Convert to Grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Step 5: Apply a Threshold
    # If a pixel changed by > 20, make it white (255). Otherwise, make it black (0).
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Step 6: Find Contours (the outlines of the white moving blobs)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 7: Analyze each moving object found
    for c in contours:
        # If the moving area is smaller than 500 pixels, skip it
        if cv2.contourArea(c) < 500:
            continue
            
        # If we reach here, it's a BIG movement! Trigger the alert.
        # Frequency: 500Hz, Duration: 100 milliseconds
        winsound.Beep(500, 100)
    
    cv2.imshow("Security Camera ", thresh)
    
    # Step 9: Update the reference frame
    # The current frame becomes the 'old' frame for the next loop iteration
    img1 = img2

    # Step 10: Check for keyboard input
    # If the user presses 'q' (ASCII 113), the loop ends
    if cv2.waitKey(15) == 113:
        break

# Clean up: Release the camera and close all windows
webcam.release()
cv2.destroyAllWindows()