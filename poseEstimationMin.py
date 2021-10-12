import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# cap = cv2.VideoCapture('PoseVideos/2.avi')
cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(f"results: {results}")
    print(f"LAND MARX: {results.pose_landmarks}")
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                              mpDraw.DrawingSpec(color=(245,66,230),  thickness=2, circle_radius=2))


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()