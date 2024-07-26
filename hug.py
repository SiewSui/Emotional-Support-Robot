#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()

# Define the threshold distance for the hand to trigger the hug action (in millimeters)
DISTANCE_THRESHOLD < 100  # Adjust this value as needed

# Initialize the motors connected to ports A and D.
#motor_left = Motor(Port.A)
motor_right_hand = Motor(Port.D)
motor_right_arm = Motor(Port.B)

# # Define the hug action.
def motor_hug():
    # Move both motors to a predefined position to simulate a hug.
    #motor_left.run_angle(500, -90, then=Stop.HOLD, wait=True)  # Rotate left motor to hug position
    motor_right_arm.run_angle(then=Stop.HOLD, wait=True, 500, -90, then=Stop.HOLD, wait=True)
    motor_right_hand.run_angle(then=Stop.HOLD, wait=True, 500, 70, then=Stop.HOLD, wait=True)  # Rotate right motor to hug position
        
    # Hold the hug position for a while.
    wait(2000)  # Wait for 2 seconds while "hugging"
        
    # Return both motors to the initial position.
    #motor_left.run_angle(500, 90, then=Stop.HOLD, wait=True)
    motor_right_hand.run_angle(500, -70, then=Stop.HOLD, wait=True)
    motor_right_arm.run_angle(500, 90, then=Stop.HOLD, wait=True)

while True:
    # Measure the distance using the Ultrasonic Sensor
    distance = ultrasonic_sensor.distance()

    # Check if the distance is below the threshold
    if distance < DISTANCE_THRESHOLD:
        motor_hug()  # Perform the hug action
        ev3.speaker.beep()  # Beep to indicate the hug action has finished
    
    # Small delay to avoid busy-waiting
    wait(100)
