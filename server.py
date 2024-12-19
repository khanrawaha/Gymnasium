
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import base64  

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

stage= None
rep_counter = 0
incorrect_count = 0
rep_stage = 'correct'



correct_rep_counter = 0
incorrect_rep_counter = 0
kickback_detected = False
incorrect_rep_detected = False
incorrect_rep_threshold = 10 

def hammercurl(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between points
            point_16 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            point_14 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            point_12 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            vector1 = [point_16[0] - point_14[0], point_16[1] - point_14[1]]
            vector2 = [point_12[0] - point_14[0], point_12[1] - point_14[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)
            angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Check if the angle is within the desired range for hammer curls


                # Repetition counting logic
            if 95.0 > angle_deg > 85.0:
                stage = "up"
            if 5.0 < angle_deg < 10.0 and stage == 'up' and rep_stage == 'correct':
                stage = "down"
                rep_counter += 1

            if 98 < angle_deg or 5.0 > angle_deg:
                rep_stage = 'incorrect'
                if stage == 'up':
                    stage = 'down'
            if 5.0 < angle_deg < 10.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            cv2.putText(image, f"correct reps : {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                            cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image

def detect_posture(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            for landmark_id, landmark in enumerate(landmarks):
                landmark_x, landmark_y, landmark_z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), landmark.z
                cv2.putText(image, f"{landmark_id}", (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                print(f"Landmark {landmark_id}: X={landmark_x}, Y={landmark_y}, Z={landmark_z:.2f}")

                point_12 = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
                point_14 = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1]),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]))
                point_16 = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1]),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]))
                cv2.line(image, point_12, point_14, (0, 255, 0), 2)
                cv2.line(image, point_14, point_16, (0, 255, 0), 2)
                cv2.line(image, point_12, point_16, (0, 255, 0), 2)

            point_16 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            point_14 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            point_12 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            vector1 = [point_16[0] - point_14[0], point_16[1] - point_14[1]]
            vector2 = [point_12[0] - point_14[0], point_12[1] - point_14[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)
            angle_deg = math.degrees(math.acos(cosine_angle))
            
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Angle between points 16, 14, and 12: {angle_deg:.2f} degrees")

            if 0 <= angle_deg <= 14:
                print("Wrist up!")
                cv2.putText(image, "Keep your wrist up!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "doing good !", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Wrist not up!")

            if 80.0 > angle_deg > 75.0:
                stage = "up"
            if 5.0 < angle_deg < 10.0 and stage == 'up' and rep_stage == 'correct':
                stage = "down"
                rep_counter += 1

            if 98 < angle_deg or 5.0 > angle_deg:
                rep_stage = 'incorrect'
                if stage == 'up':
                    stage = 'down'
            if 5.0 < angle_deg < 10.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            print(rep_counter)
            cv2.putText(image, f"correct Count: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        except Exception as e:
            print(f"Error: {e}")

    return image



def arnoldpress(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for Arnold Press exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_13[0] - point_11[0], point_13[1] - point_11[1]]
            vector2 = [point_15[0] - point_13[0], point_15[1] - point_13[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for Arnold Press exercise
            if 7 < angle_deg < 115.0 and rep_stage=="correct":
                stage = "down"
            if 0.0 < angle_deg < 6.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 125.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 0.0 < angle_deg < 6.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image


def behindtheneck(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for Behind the Neck Press exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_13[0] - point_11[0], point_13[1] - point_11[1]]
            vector2 = [point_15[0] - point_13[0], point_15[1] - point_13[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for Behind the Neck Press exercise
            if 90 < angle_deg < 170.0 and rep_stage=="correct":
                stage = "down"
            if 0.0 < angle_deg < 80.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 160.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 0.0 < angle_deg < 5.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image


def deadlift(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for deadlift exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            point_12 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            vector1 = [point_11[0] - point_12[0], point_11[1] - point_12[1]]
            vector2 = [point_13[0] - point_12[0], point_13[1] - point_12[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for deadlift exercise
            if 120.0 < angle_deg < 135.0 and rep_stage == "correct":
                stage = "down"
            if 160.0 < angle_deg < 175.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg < 70.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 160.0 < angle_deg < 175.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image

def leftcableonearmtricepsextension(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for Cable One Arm Tricep Extension exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_13[0] - point_11[0], point_13[1] - point_11[1]]
            vector2 = [point_15[0] - point_13[0], point_15[1] - point_13[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for Cable One Arm Tricep Extension exercise
            if 90 < angle_deg < 170.0 and rep_stage=="correct":
                stage = "down"
            if 0.0 < angle_deg < 80.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 104.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 0.0 < angle_deg < 5.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image



def leftshoulderfacepull(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for face pull exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_13[0] - point_11[0], point_13[1] - point_11[1]]
            vector2 = [point_15[0] - point_13[0], point_15[1] - point_13[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for face pull exercise
            if 60 < angle_deg < 130 and rep_stage == "correct":
                stage = "pull"
            if 0.0 < angle_deg < 45.0 and stage == 'pull' and rep_stage == 'correct':
                stage = "release"
                rep_counter += 1

            if angle_deg > 135.0:
                rep_stage = 'incorrect'
                if stage == 'pull':
                    stage = 'release'
            if 0.0 < angle_deg < 45.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image


def leftsidebarbellrow(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for barbell row exercise
            point_21 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_22 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_21[0] - point_22[0], point_21[1] - point_22[1]]
            vector2 = [point_23[0] - point_22[0], point_23[1] - point_22[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for barbell row exercise
            if 85.0 < angle_deg < 155.0 and rep_stage == "correct":
                stage = "down"
            if 160.0 < angle_deg < 175.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg < 70.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 160.0 < angle_deg < 175.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
    return image


def leftlunges(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for lunges exercise
            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            point_25 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            point_27 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            vector1 = [point_25[0] - point_23[0], point_25[1] - point_23[1]]
            vector2 = [point_27[0] - point_25[0], point_27[1] - point_25[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for lunges exercise
            if 70.0 < angle_deg < 90.0 and rep_stage=="correct":
                stage = "down"
            if 0.0 < angle_deg < 10.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 95.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 0.0 < angle_deg < 10.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
    return image


def rightlunges(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for right lunges exercise
            point_24 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            point_26 = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            point_28 = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            vector1 = [point_26[0] - point_24[0], point_26[1] - point_24[1]]
            vector2 = [point_28[0] - point_26[0], point_28[1] - point_26[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for right lunges exercise
            if 70.0 < angle_deg < 90.0 and rep_stage == "correct":
                stage = "down"
            if 0.0 < angle_deg < 10.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 95.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
                    # Increment only when transitioning from incorrect to correct stage
                    if rep_stage == 'correct':
                        rep_counter += 1
            if 0.0 < angle_deg < 10.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image



def platecurl(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for plate curl exercise
            point_21 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_22 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_21[0] - point_22[0], point_21[1] - point_22[1]]
            vector2 = [point_23[0] - point_22[0], point_23[1] - point_22[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for plate curl exercise
            if 110.0 < angle_deg < 120.0 and rep_stage == "correct":
                stage = "down"
            if 40.0 < angle_deg < 50.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 160.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 130.0 < angle_deg < 135.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
    return image






def pulldown(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for pull-down exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            point_12 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            vector1 = [point_11[0] - point_12[0], point_11[1] - point_12[1]]
            vector2 = [point_13[0] - point_12[0], point_13[1] - point_12[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for pull-down exercise
            if 140.0 < angle_deg < 180.0 and rep_stage == "correct":
                stage = "down"
            if 30.0 < angle_deg < 60.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg < 25.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 30.0 < angle_deg < 60.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image




def rightarmcabletricepsextension(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for Right Arm Cable Tricep Extension exercise
            point_12 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            point_14 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            point_16 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            vector1 = [point_14[0] - point_12[0], point_14[1] - point_12[1]]
            vector2 = [point_16[0] - point_14[0], point_16[1] - point_14[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for Right Arm Cable Tricep Extension exercise
            if 90 < angle_deg < 170.0 and rep_stage=="correct":
                stage = "down"
            if 0.0 < angle_deg < 80.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg > 104.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 0.0 < angle_deg < 5.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image


def righttricepsdumbellkickback(frame):
    global correct_rep_counter 
    global incorrect_rep_counter 
    global kickback_detected 
    global incorrect_rep_detected 
    global incorrect_rep_threshold 
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get landmarks for shoulders, elbows, and hands
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle between shoulders, elbows, and hands for both arms
            angle_left = math.degrees(math.atan2(left_hand[1] - left_elbow[1], left_hand[0] - left_elbow[0]))
            angle_right = math.degrees(math.atan2(right_hand[1] - right_elbow[1], right_hand[0] - right_elbow[0]))

            # Display the triceps dumbbell kickback angles on the image
            cv2.putText(image, f"Left Kickback Angle: {angle_left:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Right Kickback Angle: {angle_right:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check if the person is performing the Triceps Dumbbell Kickback exercise
            if angle_left >= 120 and angle_right >= 120:
                print("Triceps Dumbbell Kickback exercise detected!")
                cv2.putText(image, "Triceps Dumbbell Kickback Exercise Detected!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not kickback_detected:
                    kickback_detected = True
                    correct_rep_counter += 1
                    print("Correct Reps:", correct_rep_counter)
            else:
                kickback_detected = False
                if angle_left < incorrect_rep_threshold or angle_right < incorrect_rep_threshold:
                    if not incorrect_rep_detected:
                        incorrect_rep_detected = True
                        incorrect_rep_counter += 1  # Increment the incorrect_rep_counter for incorrect reps
                else:
                    incorrect_rep_detected = False
                cv2.putText(image, "Keep performing Triceps Dumbbell Kickback exercise!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

            
            cv2.putText(image, f"Correct Reps: {correct_rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2, cv2.LINE_AA)
            cv2.putText(image, f"Incorrect Reps: {incorrect_rep_counter}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image


def squats(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for deadlift exercise
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            point_12 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            vector1 = [point_11[0] - point_12[0], point_11[1] - point_12[1]]
            vector2 = [point_13[0] - point_12[0], point_13[1] - point_12[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for deadlift exercise
            if 120.0 < angle_deg < 135.0 and rep_stage == "correct":
                stage = "down"
            if 160.0 < angle_deg < 175.0 and stage == 'down' and rep_stage == 'correct':
                stage = "up"
                rep_counter += 1

            if angle_deg < 70.0:
                rep_stage = 'incorrect'
                if stage == 'down':
                    stage = 'up'
            if 160.0 < angle_deg < 175.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")


    return image



def legsextension(frame):
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    print("Received frame data in detect_posture function")
     # Check the type of the decoded data

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Calculate the angle between points for leg extension exercise
            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            point_25 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            point_27 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            vector1 = [point_25[0] - point_23[0], point_25[1] - point_23[1]]
            vector2 = [point_27[0] - point_25[0], point_27[1] - point_25[1]]

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] * 2 + vector1[1] * 2)
            magnitude2 = math.sqrt(vector2[0] * 2 + vector2[1] * 2)

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Ensure that the value passed to math.acos() falls within the valid range
            if cosine_angle >= -1 and cosine_angle <= 1:
                angle_deg = math.degrees(math.acos(cosine_angle))
            else:
                # Handle the case when the value is outside the valid range
                if cosine_angle < -1:
                    cosine_angle = -1
                elif cosine_angle > 1:
                    cosine_angle = 1
                angle_deg = math.degrees(math.acos(cosine_angle))

            # Display the angle on the image
            cv2.putText(image, f"Angle: {angle_deg:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Check if the angle is within the desired range for leg extension exercise
            if 15.0 < angle_deg < 25.0:
                stage = "up"
            if 80.0 < angle_deg < 95.0 and stage == 'up' and rep_stage == 'correct':
                stage = "down"
                rep_counter += 1

            if angle_deg < 5.0:
                rep_stage = 'incorrect'
                if stage == 'up':
                    stage = 'down'
            if 5.0 < angle_deg < 95.0 and rep_stage == 'incorrect':
                rep_stage = 'correct'
                incorrect_count += 1

            # Draw lines on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.putText(image, f"Correct reps: {rep_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Incorrect count: {incorrect_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Stage: {stage}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f"Rep stage: {rep_stage}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

    return image





@app.route('/Bicep curls', methods=['POST'])
def handle_posture_detection():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = detect_posture(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 

@app.route('/Hammer curls', methods=['POST'])
def handle_posture_detection_2():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = hammercurl(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Arnold press', methods=['POST'])
def handle_posture_detection_3():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = arnoldpress(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Behind the neck press', methods=['POST'])
def handle_posture_detection_4():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = behindtheneck(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 

@app.route('/Deadlift', methods=['POST'])
def handle_posture_detection_5():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = deadlift(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Left tricep with machine', methods=['POST'])
def handle_posture_detection_6():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = leftcableonearmtricepsextension(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Shoulder face pull', methods=['POST'])
def handle_posture_detection_7():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = leftshoulderfacepull(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 

@app.route('/Barbell row', methods=['POST'])
def handle_posture_detection_8():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = leftsidebarbellrow(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Left lunges', methods=['POST'])
def handle_posture_detection_9():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = leftlunges(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Right lunges', methods=['POST'])
def handle_posture_detection_10():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = rightlunges(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Plate curl', methods=['POST'])
def handle_posture_detection_11():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = platecurl(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 

@app.route('/pulldown', methods=['POST'])
def handle_posture_detection_12():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = pulldown(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Right tricep with machine', methods=['POST'])
def handle_posture_detection_13():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = rightarmcabletricepsextension(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 



@app.route('/Tricep kick back', methods=['POST'])
def handle_posture_detection_14():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = righttricepsdumbellkickback(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 

@app.route('/Squats', methods=['POST'])
def handle_posture_detection_15():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = squats(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 


@app.route('/Leg extension', methods=['POST'])
def handle_posture_detection_16():
    print("Incoming posture detection request") 
    frame_data = request.form['frame'] 

    decoded_frame = base64.b64decode(frame_data.split(",")[1])
    nparr = np.frombuffer(decoded_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_frame = legsextension(frame)

    desired_width = 1280  # Example - adjust as needed
    desired_height = 960  # Example - adjust as needed
    resized_image = cv2.resize(processed_frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + img_base64

    return jsonify({'processed_frame': data_uri}) 





@app.route('/reset', methods=['POST'])
def reset():
    global stage
    global rep_counter
    global incorrect_count
    global rep_stage
    
    global correct_rep_counter 
    global incorrect_rep_counter 
    global kickback_detected 
    global incorrect_rep_detected 
    global incorrect_rep_threshold 

    stage = None
    rep_counter = 0
    incorrect_count = 0
    rep_stage = 'correct'
    
    correct_rep_counter = 0
    incorrect_rep_counter = 0
    kickback_detected = False
    incorrect_rep_detected = False

    return jsonify({'status': 'reset done'})




if __name__ == '__main__':
    app.run(debug=True)



