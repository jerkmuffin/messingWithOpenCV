import re
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import relayActivator as rl
import threading as th

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

pTime = 0
cap = cv2.VideoCapture(0)
needWidthHeight = True
left_fingers_open = False
right_fingers_open= False
right_wave = False
left_wave = False
relayOn = False
reset = time.time()
wait = False
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
    if needWidthHeight:
        width = cap.get(3)
        height = cap.get(4)
        needWidthHeight = False

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
        print(f'pose problems {e}')
        pass

    try:
        landmarks = results.right_hand_landmarks.landmark
        r_index_tip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y]
        r_index_dip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].y]
        r_index_pip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].y]
        r_index_mcp = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y]
        r_middle_tip = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y]
        r_middle_dip =[landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP.value].y]
        r_middle_pip = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP.value].y]
        r_middle_mcp = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y]
        r_ring_tip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].y]
        r_ring_dip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_DIP.value].y]
        r_ring_pip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP.value].y]
        r_ring_mcp = [landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].y]
        r_pinky_tip = [landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].y]
        r_pinky_dip = [landmarks[mp_holistic.HandLandmark.PINKY_DIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_DIP.value].y]
        r_pinky_pip = [landmarks[mp_holistic.HandLandmark.PINKY_PIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_PIP.value].y]
        r_pinky_mcp = [landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].y]

        if (r_index_tip[1] < r_index_dip[1] < r_index_pip[1] < r_index_mcp[1]) and \
            (r_middle_tip[1] < r_middle_dip[1] < r_middle_pip[1] < r_middle_mcp[1]) and \
            (r_ring_tip[1] < r_ring_dip[1] < r_ring_pip[1] < r_ring_mcp[1]) and \
            (r_pinky_tip[1] < r_pinky_dip[1] < r_pinky_pip[1] < r_pinky_mcp[1]):
              right_fingers_open = True
        else:
              right_fingers_open = False
    except Exception as e:
        print(f'right hand problems {e}')
        right_fingers_open = False
        pass

    try:
        landmarks = results.left_hand_landmarks.landmark
        l_index_tip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP.value].y]
        l_index_dip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].y]
        l_index_pip = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].y]
        l_index_mcp = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y]
        l_middle_tip = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y]
        l_middle_dip =[landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP.value].y]
        l_middle_pip = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP.value].y]
        l_middle_mcp = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP.value].y]
        l_ring_tip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP.value].y]
        l_ring_dip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_DIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_DIP.value].y]
        l_ring_pip = [landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP.value].y]
        l_ring_mcp = [landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].x, landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP.value].y]
        l_pinky_tip = [landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_TIP.value].y]
        l_pinky_dip = [landmarks[mp_holistic.HandLandmark.PINKY_DIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_DIP.value].y]
        l_pinky_pip = [landmarks[mp_holistic.HandLandmark.PINKY_PIP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_PIP.value].y]
        l_pinky_mcp = [landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].x, landmarks[mp_holistic.HandLandmark.PINKY_MCP.value].y]

        if (l_index_tip[1] < l_index_dip[1] < l_index_pip[1] < l_index_mcp[1]) and \
          (l_middle_tip[1] < l_middle_dip[1] < l_middle_pip[1] < l_middle_mcp[1]) and \
          (l_ring_tip[1] < l_ring_dip[1] < l_ring_pip[1] < l_ring_mcp[1]) and \
          (l_pinky_tip[1] < l_pinky_dip[1] < l_pinky_pip[1] < l_pinky_mcp[1]):
            left_fingers_open = True
        else:
            left_fingers_open = False
    except Exception as e:
        print(f'left hand problems {e}')
        left_fingers_open = False
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
        cv2.putText(frame, str(l_angle), tuple(np.multiply(elbow_flip(l_elbow), [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # print right elbow angle
    if 'r_angle' in globals():
        cv2.putText(frame, str(r_angle), tuple(np.multiply(elbow_flip(r_elbow), [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if left_fingers_open and (l_angle > 30 and l_angle < 100):
        left_wave = True
    else:
        left_wave = False


    if right_fingers_open and (r_angle > 30 and r_angle < 100):
        right_wave = True
    else:
        right_wave = False

    print(f"right wave: {right_wave}\nleft wave: {left_wave}\nrelay on: {relayOn}")
    shapes = np.zeros_like(frame, np.uint8)


    if relayOn:
        cv2.rectangle(shapes, (0, 0), (640, 480), (0, 0, 255), cv2.FILLED)
    elif left_wave or right_wave and not relayOn:
        cv2.rectangle(shapes, (0, 0), (640, 480), (0, 255, 0), cv2.FILLED)
    
    out = frame.copy()
    out = cv2.addWeighted(frame, 1, shapes, .5, 0)
    cv2.imshow('MediaPipe Holistic', out)

    def turnOffCB():
        rl.off()
        global relayOn
        relayOn = False

    def waitedTurnOnCB():
        rl.on()
        global relayOn
        relayOn = True
        timer = th.Timer(.25, turnOffCB)
        timer.start()

    def activateRelay():
        timer = th.Timer(.5, waitedTurnOnCB)
        timer.start()
        
        print('started timer......')


    if right_wave or left_wave:
        now = time.time()
        if now > reset+3:
            activateRelay()
            reset = time.time()


    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
