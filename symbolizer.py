import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import sys
import argparse
from glob import glob

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

file = 0
cap = cv2.VideoCapture(file)
model_keypoints = mp_hands.Hands(min_detection_confidence=0.55,
                                 min_tracking_confidence=0.55,
                                 max_num_hands=2)
#model_classify = keras.models.load_model("hand_models/modelv4.9")
# # TODO: Find a way to input the class names rather than hardcoding them
"""class_names = sign_names = [
    "ANIMAL",
    "BAD",
    "BENT FLAT",
    "BENT GUN",
    "BENT TWO",
    "CLAW",
    "CLOSED",
    "CUP",
    "EIGHT",
    "ELEVEN",
    "FIST",
    "FIVE",
    "FLAT",
    "FLAT OKAY",
    "FLAT ROUND",
    "FLICK",
    "FOUR",
    "GOOD",
    "GUN",
    "HOOK",
    "I-LOVE-YOU",
    "KEY",
    "LETTER-C",
    "LETTER-M",
    "MIDDLE",
    "OKAY",
    "ONE-HAND-LETTER-D",
    "ONE-HAND-LETTER-K",
    "OPEN SPOON",
    "PLANE",
    "POINT",
    "ROUND",
    "RUDE",
    "SMALL",
    "SPOON",
    "THICK",
    "THREE",
    "TWELVE",
    "TWO",
    "WISH",
    "WRITE"
]"""

symbols = {"FLAT":"A", "FIVE":"B", "CLOSED":"C"}

handshape_tracked_symbols = {"Right":[], "Left":[], "Frame":[], "Label":[]}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Rescaling image to 800x... resolution
    scale_percent = 1/(image.shape[1] / 800)
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    image = cv2.resize(cv2.flip(image, 1), dim)

    # Getting handpose estimates
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model_keypoints.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks is not None:
        # For each hand we want to look at the keypoints and grab a ratio
        # To normalize them

        for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = results.multi_hand_landmarks[hand_landmarks_idx].landmark._values
            kp = []
            ratio = image.shape[1] / image.shape[0]
            for point in keypoints:
                kp.append(np.array([point.x * ratio, point.y, point.z * ratio]))

            label = results.multi_handedness[hand_landmarks_idx].classification._values[0].label

            keypoints = np.array(kp)
            keypoints = subtract_offset(keypoints)
            keypoints = rotate_3d(keypoints)
            keypoints = normalise(keypoints)

            if label == "Left":
                keypoints[:, 0] *= -1

                keypoints = tf.expand_dims(keypoints.flatten(), 0)
                predictions = model_classify.predict(keypoints)
                score = predictions[0]

                class_label = class_names[np.argmax(score)]
                class_score = str(round(100 * np.max(score), 2))

            else:

                keypoints = tf.expand_dims(keypoints.flatten(), 0)
                predictions = model_classify.predict(keypoints)
                score = predictions[0]

                class_label = class_names[np.argmax(score)]
                class_score = str(round(100 * np.max(score), 2))

            handshape_tracked_symbols[label].append(symbols[class_label])
            handshape_tracked_symbols["Frame"].append(str(frame_count))
            # Can use the ground truth files to see signing and non-signing stuff
            handshape_tracked_symbols["Label"].append("-1")
    else:
        # We want to assign a symbol to this frame. This is like the "H" symbol
        # that we assigned to the frame where we declared we couldn't detect the
        # hand in the location symbolizer

        handshape_tracked_symbols["Right"].append("H")
        handshape_tracked_symbols["Left"].append("H")
        handshape_tracked_symbols["Frame"].append(str(frame_count))
        # Can use the ground truth files to see signing and non-signing stuff
        handshape_tracked_symbols["Label"].append("-1")





    cv2.imshow("Window", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
