#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import cv2
import numpy as np
from deepface import DeepFace
import math
import os
import pyttsx3
import random
import platform
import speech_recognition as sr
import pyaudio
from groq import Groq

# Create your objects here.
ev3 = EV3Brick()

# Write your program here.
ev3.speaker.beep()

# Open audio cue text file
try:
    with open('audiocue.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        audio_cues = [line.strip(' \t\r\n') for line in lines if line.strip(' \t\r\n')]

    if audio_cues:  # Check if the list is not empty
         # Extract specific cues from the list
        hug_cues = audio_cues[0:3]
        quote_cues = audio_cues[3:14]
        conversational_cue = audio_cues[14:17]
        affirmative_responses = audio_cues[17:24]
        negative_responses = audio_cues[24:]
    else:
        print("No audio cues found in the file.")
except FileNotFoundError:
    print("The audiocue.txt file was not found.")

# Initialize pyttsx3
engine = pyttsx3.init()

#Initalize sr
ls = sr.Recognizer()

# Set your API key
GROQ_API_KEY = "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"

# Initialize the Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)

def seekconsent():
    case = input(">>> ")

    if case.lower() in affirmative_responses:
        return "2"
    else:
        speak("Ah! I see, is there anything you would like to talk about instead?")
        case = input(">>> ")
        if case.lower() in affirmative_responses:
            return "1"
        else:
            speak("Very well then, just know that if you need anything I'll be here.")
            return "3"  # Return 3 to indicate end of conversation

def speak(robot_message):
    engine.say(robot_message)
    engine.runAndWait()

def conversation(history, user_message):
    history.append({"role": "user", "content": user_message})
    chat = client.chat.completions.create(
        messages=history,
        model="llama3-8b-8192",
        max_tokens=150,  # Adjust this to limit the length of the response
        temperature=0.7  # Adjust this to control randomness in the response
    )
    response = chat.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    return response

# To put into a text file:
def remember():
    print()

# MacOS specific initialization for pyttsx3
if platform.system() == 'Darwin':
    try:
        import Foundation
        import AppKit
    except ImportError:
        import objc

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Initialize video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('nodcontrol.avi', fourcc, 20.0, (640, 480))

# Distance function
def distance(x, y):
    return math.sqrt((x[0] - y[0]) * 2 + (x[1] - y[1]) * 2)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Path to face cascade
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.isfile(cascade_path):
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Capture source video
cap = cv2.VideoCapture(0)

# Function to get coordinates
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except:
        return int(p1[0][0]), int(p1[0][1])

# Define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX

# Define movement thresholds
gesture_threshold = 175

# Initialize face tracking
face_found = False
emotion_detected = False
while not face_found:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_center = (x + w // 2, y + h // 2)  # Initial tracking point on the face center
        face_found = True
    cv2.imshow('image', frame)
    out.write(frame)
    cv2.waitKey(1)

p0 = np.array([[face_center]], np.float32)
gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # Number of frames a gesture is shown

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Emotion recognition
    if not emotion_detected:
        result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        if len(result) > 0:
            emotion = result[0]['dominant_emotion']
            txt = str(emotion)
            cv2.putText(frame, txt, (50, 150), font, 1, (0, 255, 0), 3)
            cv2.imshow('image', frame)
            out.write(frame)
            cv2.waitKey(1)

            # Play a random text-to-speech cue if the emotion is sad
            if emotion == 'sad':
                hug_cue = random.choice(hug_cues)
                engine.say(hug_cue)
                engine.runAndWait()
                emotion_detected = True
                break  # Break the loop to start gesture detection

# cv2.destroyAllWindows()
# if we don't close the camera & open it agn seems faster ohh

# Start head gesture detection after emotion detection
if emotion_detected:
    # cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face again to correct the drift
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center = (x + w // 2, y + h // 2)
            p0 = np.array([[face_center]], np.float32)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st[0][0] == 1:
            cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
            cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

            # Get the xy coordinates for points p0 and p1
            a, b = get_coords(p0), get_coords(p1)
            x_movement += abs(a[0] - b[0])
            y_movement += abs(a[1] - b[1])

            text = 'x_movement: ' + str(x_movement)
            if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
            text = 'y_movement: ' + str(y_movement)
            if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (0, 0, 255), 2)

            if x_movement > gesture_threshold:
                gesture = 'No'
            if y_movement > gesture_threshold:
                gesture = 'Yes'
            if gesture and gesture_show > 0:
                cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 1.2, (0, 0, 255), 3)
                gesture_show -= 1
            if gesture_show == 0:
                if gesture == 'Yes':
                    # Start the conversation with a random initial prompt
                    message = random.choice(conversational_cue)
                    speak(message)
                    print(message)

                    # Initialize the conversation history
                    instruction = "You are a friendly assistant name Ajex. Be concise, use simple language, and encourage the user to talk about their problems. If you detect something serious in the user, or something that is deeply bothering them, either encourage them to share more about it."
                    history = [{"role": "system", "content": instruction}]
                    user_message = input(">>> ")

                    print("Listening: ")
                    with sr.Microphone() as AudioSource:
                        ls.adjust_for_ambient_noise(AudioSource, 10)
                        user_speech = ls.listen(AudioSource)
                        try:
                            user_message = ls.recognize_google(user_speech)
                            response = conversation(history, user_message)
                            print(response)
                            speak(response)
                            break
                        except:
                            speak("I'm truly sorry but I did not quite get that. Can you run that by me again?")
                            break
                    
                    quote_cue = random.choice(quote_cues)
                    engine.say(quote_cue)
                    engine.say('Feel free to grab a snack!')
                    engine.runAndWait()
                    break
                else: 
                    engine.say('Alright! Have nice day! Feel free to grab a snack!')
                    engine.runAndWait()
                    emotion_detected = False  # Reset emotion detection
                    break  # Go back to emotion detection

            # Update the previous points with the current points
            # p0 = p1

        cv2.imshow('image', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    # out.release()
