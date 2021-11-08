import cv2
import mediapipe as mp
import numpy as np
import time

### https://morioh.com/p/e1b81be4bb0f
### thanks for the help
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# cap = cv2.VideoCapture('PoseVideos/2.avi')
cap = cv2.VideoCapture(0)
pTime = 0

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgGRY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = pose.process(imgRGB)

    try:
        landmarks = results.pose_landmarks.landmark
        l_shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        r_shoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    except Exception as e:
        print(f'problems {e}')
        pass

    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                              mpDraw.DrawingSpec(color=(245,66,230),  thickness=2, circle_radius=2))


    cTime = time.time()
    fps = 1/(cTime-pTime)
    fps = f"fps: {int(fps)}"
    pTime = cTime

    if 'l_angle' in globals():
        cv2.putText(img, str(l_angle), tuple(np.multiply(l_elbow, [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if 'r_angle' in globals():
        cv2.putText(img, str(r_angle), tuple(np.multiply(r_elbow, [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str((fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()