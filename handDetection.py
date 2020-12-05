import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# For webcam input:
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5,)
cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)


i =0
while cap.isOpened():
  success, image = cap.read()
  frame = image
  w, he, c = frame.shape
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)
  # print(results)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    i = i+1
    for hand_landmarks in results.multi_hand_landmarks:
      print(i)
      h = hand_landmarks.landmark[8]

      a = float(h.x)
      b = float(h.y)
      x_px = min(math.floor(a * w), w - 1)
      y_px = min(math.floor(b * he), he - 1)
      print(x_px, y_px)
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imshow('MediaPipe Hands', image)
  cv2.imshow('orginal', frame)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()