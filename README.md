# Autonomous-Robot-Math-Solver
Robot autonomously analyzes a math problem on a whiteboard and physically writes out the answer using long addition and long multiplication. Program utilizes computer vision to analyze whiteboard and implements an inverse kinematics algorithm to move the robot arm and draw numbers.

## Team Members

## Project Description
The goal of this project is to present a robot with a math problem on a white-board (currently just addition of any integer size within camera frame) and have the robot emulate the actions that a human would. This involves the robot solving single-digit sums from right to left across the problem, including carrying the 1 and physically marking the carry when there is overflow in the summation. The robot writes the proper calculation of the math problem below the actual problem on the whiteboard.
This project requires a few main parts: 
1. The computer first utilized the camera on the front to read an image of the math problem on the white board. This was then passed through a computer vision program that masked the color of the written problem (blue in our case) and broke apart the digits. It then used a convolutional neural network to classify the individual numbers for the calculation. This portion would return the digit value and pixel location of each digit in a ordered list (left to right and top to bottom).
2. The actual mathematics was the simple part: breaking apart the integers to sum one column at a time and passing that through a control flow in the program.
3. The other portion of the program took in an input from the digit classifier as to where the robot should start drawing. An inverse kinematics algorithm was used to adjust the robot arm ending point's x, y, and z value to move to the starting drawing portion.
4. The robot then had to implement another inverse kinematics algorithm to actually draw. The robot can currently draw horizontal and vertical straight lines. All digits drawn are combinations of the horizontal and vertical movements.


## System Architecture
We implemented algorithms in two major robotics areas: computer vision and inverse kinematics. The computer vision portion analyzes images of an addition problem. The kinematics component directs the arm to draw the answer digits in their correct locations. We highlight these components below.

### Computer Vision
The computer vision segment analyzes handwritten addition problems using the Turtlebot's camera. The function `extract_digits()` in `extract_digits.py` manages the image segmentation and classification. We assume that the problem's digits are written in blue marker; these lines are the only blue colors in the visual frame. We first threshold the image to extract these blue lines and then fit bounding boxes for each digit. Then, we clip the image to each bounding box, convert the clipping to grayscale, and classify the digit as 0-9 using a convolutional neural network trained on MNIST (`DigitClassifier.predict()` in `extract_digits.py`). Once we have the classified digits, we group them into two horizontal rows. This grouping works by splitting along the largest vertical gap amongst digits (`group_digits()` in `extract_digits.py`). We find that the camera is low resolution and sometimes inconsistent. To mitigate transient issues, we take multiple trials of the image segmentation and use the majority classification results. Once we have the final digit results, we perform the addition (`image_callback()` in `arm.py`) to get the digits to draw.

### Inverse Kinematics
The inverse kinematics component controls the arm's movements when drawing. The first portion of this process is to find *where* to draw the digits. We make this determination using the location of the problem in the image (`get_answer_locations()` in `utils.py`). This process produces the locations to draw in pixels. We convert these pixel locations to distances in meters using hand-calibrated measurements of the visible frame (`image_point_to_draw_digit()` in `utils.py`). We then have the positions in meters of where to draw, and we manually create functions to handle the drawing of each digit (`draw_[digit]` in `arm.py`). We move the arm to these desired locations using inverse kinematics by treating the arm as a 3-joint entity; we always set the arm's joint angle `3` to `0`. Consider moving to a point `(horizontal, vertical)` on the board. Then, `theta0`, the angle for joint `0`, is `theta0 = arctan2(horizontal, length2 + length3 + pen_length)` where `length2` is the length between joints `1` and `2` and `length3` is the length between joints `2`, and `3`. We want to arm to touch the board at the `vertical` position along this `theta0`. Viewed from a horizontal cross-section, this problem is analogous to the 2-joint arm from lecture ten where `x = horizontal / sin(theta0)` and `y = vertical`. We can thus use the corresponding inverse kinematics to solve for `theta1` and `theta2`. The function `set_arm_position_vertical()` in `arm.py` implements this logic. The Turtlebot uses these kinematics to draw the answer and carry digits from least to most significant (`draw_answer_digit()` in `arm.py`).

## ROS Node Diagram

## Execution
Mention: digits should be in blue, other parts in another color; digits should be spaced apart for proper segmentation--the problem is that the camera is low resolution; the exact distance between the robot and the board.

## Challenges
(1) Originally tried to extract digits using grayscale and assuming it was the only thing drawn on the board, but lighting/glare on board makes this difficult
(2) Joint 2 is set slightly ahead of Joint 1. This throws distances and angles off in the inverse kinematics calculations.

## Future Work
Add more problems types (needs higher set camera for more usuable space on board), handle arbitrary colors on board (better resolution camera), and have turtlebot drive up to the board to start drawing (accurate LiDAR).

## Takeaways

