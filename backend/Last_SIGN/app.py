from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Global variable to store the detected gesture
detected_gesture = "None"

# Function to process video frames
def generate_frames():
    global detected_gesture
    cap = cv2.VideoCapture(0)  # Use webcam
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    lm_list = []
                    for id, lm in enumerate(hand_landmark.landmark):
                        lm_list.append(lm)

                    finger_fold_status = []
                    for tip in finger_tips:
                        if lm_list[tip].x < lm_list[tip - 2].x:
                            finger_fold_status.append(True)
                        else:
                            finger_fold_status.append(False)

                    # Gesture Detection Logic
                    if lm_list[3].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "OK"
                        cv2.putText(img, "OK!!!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # ONE
                    if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[12].y:
                        detected_gesture = "ONE"
                        cv2.putText(img, "ONE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # TWO
                    if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "TWO"
                        cv2.putText(img, "TWO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # THREE
                    if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "THREE"
                        cv2.putText(img, "THREE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # FOUR
                    if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                        detected_gesture = "FOUR"
                        cv2.putText(img, "FOUR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # FIVE
                    if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < \
                            lm_list[5].x:
                        detected_gesture = "FIVE"
                        cv2.putText(img, "FIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # SIX
                    if lm_list[4].y < lm_list[3].y and lm_list[20].y < lm_list[18].y and lm_list[8].y > lm_list[6].y and \
                            lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y:
                        detected_gesture = "SIX"
                        cv2.putText(img, "SIX", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # SEVEN
                    if lm_list[4].y < lm_list[3].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "SEVEN"
                        cv2.putText(img, "SEVEN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # EIGHT
                    if lm_list[4].y < lm_list[3].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "EIGHT"
                        cv2.putText(img, "EIGHT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # NINE
                    if lm_list[4].y < lm_list[3].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "NINE"
                        cv2.putText(img, "NINE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # A
                    if lm_list[4].y > lm_list[3].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "A"
                        cv2.putText(img, "A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # B
                    if lm_list[4].y > lm_list[3].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                            lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
                        detected_gesture = "B"
                        cv2.putText(img, "B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # C
                    if lm_list[4].y < lm_list[3].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                            lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                        detected_gesture = "C"
                        cv2.putText(img, "C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # D
                    if lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                            lm_list[20].y > lm_list[18].y and abs(lm_list[4].x - lm_list[8].x) < 0.05 and abs(lm_list[4].y - lm_list[8].y) < 0.05:
                        detected_gesture = "D"
                        cv2.putText(img, "D", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # E
                    if lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                            lm_list[20].y > lm_list[18].y and lm_list[4].y > lm_list[3].y:
                        detected_gesture = "E"
                        cv2.putText(img, "E", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    mp_draw.draw_landmarks(img, hand_landmark,
                                           mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                           mp_draw.DrawingSpec((0, 255, 0), 4, 2))

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    global detected_gesture
    return jsonify({"gesture": detected_gesture})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3030,debug=True)