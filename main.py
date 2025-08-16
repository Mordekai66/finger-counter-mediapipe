import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    finger_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            #* list of (x, y) points for each finger landmark
            landmarks = [(int(p.x * width), int(p.y * height)) for p in hand_landmarks.landmark]

            hand_label = handedness.classification[0].label  #* 'Right' or 'Left'

            #* Thumb
            thumb_tip_x, thumb_ip_x = landmarks[4][0], landmarks[3][0]
            if hand_label == "Right" and thumb_tip_x < thumb_ip_x:
                finger_count += 1
            elif hand_label == "Left" and thumb_tip_x > thumb_ip_x:
                finger_count += 1

            #* Other fingers
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            for tip, pip in zip(finger_tips, finger_pips):
                if landmarks[tip][1] < landmarks[pip][1]:
                    finger_count += 1

            cv2.putText(frame, f'Fingers: {finger_count}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()