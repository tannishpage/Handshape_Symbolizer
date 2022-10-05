from tabnanny import verbose
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import sys
import argparse
from glob import glob
from dependencies.rotate import *
import time

opposite = lambda x: "Right" if x == "Left" else "Left"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
folder = "/home/aravind/mtpt/Videos/Julie_Videos"
files = [f for f in os.listdir(folder)] #"/home/tannishpage/Videos/Filtered_For_Perfection/PEXbQEnjzo8_Filtered_OLD.mp4" #"/home/tannishpage/Videos/Auslan_Videos_JKorte/Person_214.mp4"

model_keypoints = mp_hands.Hands(min_detection_confidence=0.55,
                                 min_tracking_confidence=0.55,
                                 max_num_hands=2)
model_classify = keras.models.load_model("../A-Method-for-Clustering-and-Classifying-Hand-shapes-from-Auslan/hand_models/modelv5.1_Good_model")
# # TODO: Find a way to input the class names rather than hardcoding them
class_names = sign_names = ['WRITE', 'CLAWED FOUR', 'LETTER-M', 'FLICK',
                            'EIGHT', 'OKAY', 'PLANE', 'ONE-HAND-LETTER-D',
                            'I-LOVE-YOU', 'KEY', 'LETTER-C', 'RUDE',
                            'CLAWED EIGHT', 'ONE-HAND-LETTER-K', 'CLOSED',
                            'FOUR', 'BENT GOOD', 'BAD', 'GOOD', 'TWELVE',
                            'TWO', 'NINE', 'THICK', 'CUP', 'THREE', 'FIVE',
                            'SMALL', 'ANIMAL', 'FLAT ROUND', 'FLAT',
                            'CLAWED NINE', 'FIST', 'GUN', 'BENT THREE',
                            'BENT GUN', 'POINT', 'BENT BAD', 'SPOON', 'ELEVEN',
                            'OPEN SPOON', 'CLAW', 'HOOK', 'BAD LUCK',
                            'BENT FLAT', 'MIDDLE', 'BENT TWO', 'ROUND',
                            'FLAT OKAY', 'WISH']

symbols = {sign:i for i, sign in enumerate(sign_names)}
for file in files:
    start = time.time()
    print("Processing", file)
    cap = cv2.VideoCapture(os.path.join(folder, file))
    handshape_tracked_symbols = {"Right":[], "Left":[], "Frame":[], "Label":[]}
    frame_count = 0
    left = 0
    right = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count < total_frames:
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        percentage = (frame_count/total_frames)*100
        rate = frame_count / (time.time() - start)
        sys.stdout.write("\r[{}{}] {:.2f}% {}/{}\t{:.2f} fps".format('='*int(percentage/2),
                                                    '.' *(50 - int(percentage/2)),
                                                    percentage, frame_count,
                                                    total_frames, rate))
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

        found_left = False
        found_right = False

        if results.multi_hand_landmarks is not None:
            # For each hand we want to look at the keypoints and grab a ratio
            # To normalize them
            for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = results.multi_hand_landmarks[hand_landmarks_idx].landmark
                kp = []
                ratio = image.shape[1] / image.shape[0]
                for point in keypoints:
                    kp.append(np.array([point.x * ratio, point.y, point.z * ratio]))

                label = results.multi_handedness[hand_landmarks_idx].classification[0].label
                if label == "Right" and found_right:
                    print("\nFound another Right")
                    label = opposite(label)
                
                if label == "Left" and found_left:
                    print("\nFound another Left")
                    label = opposite(label)

                #print(results.multi_handedness[hand_landmarks_idx].classification[0].index, results.multi_handedness[hand_landmarks_idx].classification[0].label, results.multi_handedness[hand_landmarks_idx].classification[0].score)

                keypoints = np.array(kp)
                keypoints = subtract_offset(keypoints)
                keypoints = rotate_3d(keypoints)
                keypoints = normalise(keypoints)
                #print(label)
                if label == "Left":
                    left += 1
                    keypoints[:, 0] *= -1

                    keypoints = tf.expand_dims(keypoints.flatten(), 0)
                    predictions = model_classify.predict(keypoints, verbose=0)
                    score = predictions[0]

                    class_label = class_names[np.argmax(score)]
                    class_score = str(round(100 * np.max(score), 2))
                    found_left = True

                else:
                    right += 1
                    keypoints = tf.expand_dims(keypoints.flatten(), 0)
                    predictions = model_classify.predict(keypoints, verbose=0)
                    score = predictions[0]

                    class_label = class_names[np.argmax(score)]
                    class_score = str(round(100 * np.max(score), 2))
                    found_right = True

                handshape_tracked_symbols[label].append(symbols[class_label])
                handshape_tracked_symbols["Frame"].append(str(frame_count))
                # Can use the ground truth files to see signing and non-signing stuff
                handshape_tracked_symbols["Label"].append("-1")

                if len(results.multi_hand_landmarks) < 2:
                    # Setting the opposite hand to "H" because the model couldn't
                    # see it
                    handshape_tracked_symbols[opposite(label)].append("H")
                    if opposite(label) == "Left":
                        left += 1
                    else:
                        right += 1
            #print(left, right, left == right, len(handshape_tracked_symbols["Left"]) == len(handshape_tracked_symbols["Right"]),  sep=", ")
        else:
            # We want to assign a symbol to this frame. This is like the "H" symbol
            # that we assigned to the frame where we declared we couldn't detect the
            # hand in the location symbolizer

            handshape_tracked_symbols["Right"].append("H")
            handshape_tracked_symbols["Left"].append("H")
            handshape_tracked_symbols["Frame"].append(str(frame_count))
            # Can use the ground truth files to see signing and non-signing stuff
            handshape_tracked_symbols["Label"].append("-1")





        #cv2.imshow("Window", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    #print(len(handshape_tracked_symbols["Left"]), len(handshape_tracked_symbols["Right"]))
    symbol_file = open(f"./{file}.txt", 'w')
    symbol_file.write(f"frame:{','.join(handshape_tracked_symbols['Frame'])}\nleft:{','.join(handshape_tracked_symbols['Left'])}\nright:{','.join(handshape_tracked_symbols['Right'])}\nlabel:{','.join(handshape_tracked_symbols['Label'])}")
    symbol_file.close()
    print(f"\nProcessing time {time.time() - start:.2f} seconds")
