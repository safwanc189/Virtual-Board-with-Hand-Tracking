import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands, min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, hand_num=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

    def fingers_up(self, lm_list):
        if lm_list:
            fingers = []

            # Thumb
            if lm_list[4][1] > lm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lm_list[4 * id][2] < lm_list[4 * id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
        return [0, 0, 0, 0, 0]

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        if lm_list:
            print(lm_list)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
