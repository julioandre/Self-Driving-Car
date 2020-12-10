# Self-Driving-Car
This is an attempted spyder python project which uses deep Q-learning to implement a virtual self driving car. 
The concept of this is to use Artificial Neural Networks to Implement a virtual self driving car using a pre-created environment.

# Contents of the repository
The repository contains just one branch that contains all the necessary files for the project.

# Folder/ File Structure
The repo contains one folder in the folder there are 4 files
1. ai.py: This contains the code for the virtual car. This code implements features of Artificial Neural Networks eg. Decision Process, Back Propagation etc. this makes up the main 
  part of the learning process of our virtual car

2. car.kv: This file contains all the code to create our car

3. last_brain.pth: This file is where the 100 previous decisions rewards and actions taken by the virtual car is stored.

4. map.py: This file is the code to create and setup our virtual environment for the car.

# Starting the project

The project can be started by opening the folder in a Spyder IDE and then open your map.py file, hit the command Ctrl+A and then enter.
This will start the virtual environment with the virtual car.
The virtual car has two destinations set:
 - Top left corner of the screen
 - Bottom Right Corner of the screen
 Aim of the program is to get the car to these destination. Finally with the mouse we can draw sand which is depicted in yellow lines on the environment which serve as roads for the car
 and the AI will learn to not cross these lines as it shows they are going off road
 
 # Tech stack and dependencies
 The Project is built on
  - Python
  - TensorFlow
  - Numpy
  - PyTorch
 
 





