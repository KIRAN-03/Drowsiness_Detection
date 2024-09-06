from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils      #image utilities
import dlib       #Dougherty Library (facial recognition, object detection, image segmentation)
import cv2

# Initialize mixer and load alert sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and frame check for EAR
thresh = 0.25
frame_check = 20

# Initialize face detector and predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get indexes for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize counters
flag = 0
total_frames = 0
true_positives = 0
false_negatives = 0
false_positives = 0
true_negatives = 0

# Main loop
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)   #converts to numpy array for calculation
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Increment total frames
        total_frames += 1
        
        # Check if EAR is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:    #drowsiness detected
                cv2.putText(frame, "****************ALERT!****************", (10, 30),		
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                true_positives += 1
                mixer.music.play()
            else:
                false_positives += 1
        else:
            if flag >= frame_check:         # drowsiness not detected
                false_negatives += 1
            else:
                true_negatives += 1     
            flag = 0
    
    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF      #0xFF- hexadecimal(11111111)
    
    # Break the loop with 'q' key
    if key == ord("q"):    #returns ascii of q
        break

# Print accuracy metrics
accuracy = (true_positives + true_negatives) / total_frames
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Total frames: {total_frames}")
#print(f"True positives: {true_positives}")
#print(f"False positives: {false_positives}")
#print(f"True negatives: {true_negatives}")
#print(f"False negatives: {false_negatives}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Cleanup
cv2.destroyAllWindows()    #closes all windows and deallocates memory
cap.release()      #releases the video capturing
