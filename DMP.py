import cv2
import dlib
import serial
import time
from scipy.spatial import distance as dist
from imutils import face_utils

# SERIAL
arduino = serial.Serial("COM8", 9600)
time.sleep(2)

# FUNCTIONS
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (2.0 * D)

# SETTINGS
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.75

WARNING_TIME = 10
SLEEP_TIME = 20

sleep_start_time = None

# LOAD MODEL
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # markings
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255,255,0), 1)

    
        # DROWSINESS (TIME BASED)
        
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        elif ear < EAR_THRESHOLD:

            if sleep_start_time is None:
                sleep_start_time = time.time()

            elapsed = time.time() - sleep_start_time

            cv2.putText(frame, f"Sleep Time: {int(elapsed)}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Warning
            if elapsed >= WARNING_TIME and elapsed < SLEEP_TIME:
               remaining = int(20 - elapsed)
               cv2.putText(frame, "WARNING", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
               cv2.putText(frame, f"Slowing in: {remaining}s", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
               arduino.write(b'1')   

            #while slept
            elif elapsed >= SLEEP_TIME:
                cv2.putText(frame, "SLEEP DETECTED!", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                arduino.write(b'2')   #motor
            else:
                sleep_start_time = None

                
        else:
            arduino.write(b'3')   # NORMAL

    cv2.imshow("Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
