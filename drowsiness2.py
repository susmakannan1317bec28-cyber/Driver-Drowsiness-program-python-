import cv2
import dlib
import serial
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

# ARDUINO CONNECTION
arduino = serial.Serial("COM7", 9600)
time.sleep(2)

# -------- SIGNAL CONTROL --------
last_signal = None

def send_signal(signal):
    global last_signal
    if signal != last_signal:
        arduino.write(signal.encode())
        print("Sent:", signal)
        last_signal = signal

# -------- FUNCTIONS --------
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

# -------- THRESHOLDS --------
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75
WARNING_TIME = 10
SLEEP_TIME = 20

# -------- LOAD MODELS --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# -------- CAMERA --------
cap = cv2.VideoCapture(0)
sleep_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        send_signal('3')  # No face → assume normal
        sleep_start_time = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 255, 0), 1)

        # -------- DROWSINESS --------
        if ear < EAR_THRESHOLD:
            if sleep_start_time is None:
                sleep_start_time = time.time()

            elapsed = time.time() - sleep_start_time

            cv2.putText(frame, f"Sleep Time: {int(elapsed)}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # WARNING
            if WARNING_TIME <= elapsed < SLEEP_TIME:
                remaining = int(SLEEP_TIME - elapsed)
                cv2.putText(frame, "WARNING", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Slowing in: {remaining}s", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                send_signal('1')

            # SLEEP
            elif elapsed >= SLEEP_TIME:
                cv2.putText(frame, "SLEEP DETECTED!", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                send_signal('2')

        else:
            sleep_start_time = None

            # YAWNING
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                send_signal('0')

            else:
                send_signal('3')  # NORMAL

    cv2.imshow("Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)  # stability

cap.release()
cv2.destroyAllWindows()
arduino.close()
