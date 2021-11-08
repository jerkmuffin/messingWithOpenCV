import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

pTime = 0
cap = cv2.VideoCapture(0)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def elbow_flip(elbow):
    x = elbow[0]
    flipped = np.array([1-x, elbow[1]])
    return flipped

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    try:
        landmarks = results.pose_landmarks.landmark
        l_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        r_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    except Exception as e:
        print(f'problems {e}')
        pass

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style()
        )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())

    cTime = time.time()
    fps = 1/(cTime-pTime)
    fps = f"fps: {int(fps)}"
    pTime = cTime

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(image, 1)

    # print FPS
    cv2.putText(frame, str((fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # print left elbow angle 
    if 'l_angle' in globals():
        cv2.putText(frame, str(l_angle), tuple(np.multiply(elbow_flip(l_elbow), [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # print right elbow angle
    if 'r_angle' in globals():
        cv2.putText(frame, str(r_angle), tuple(np.multiply(elbow_flip(r_elbow), [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('MediaPipe Holistic', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()