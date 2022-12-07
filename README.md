# Autonomous-Robot-Math-Solver
Robot autonomously analyzes a math problem on a whiteboard and physically writes out the answer using long addition and long multiplication. Program utilizes computer vision to analyze whiteboard and implements an inverse kinematics algorithm to move the robot arm and draw numbers.

## Team Members

## Project Description

## System Architecture
We implemented algorithms in two major robotics areas: computer vision and inverse kinematics. The computer vision portion analyzes images of an addition problem. The kinematics component directs the arm to draw the answer digits in their correct locations. We highlight each of these components in detail below.

### Computer Vision
The computer vision segment handles the analysis of addition problems using the Turtlebot's camera. The function `extract_digits()` in `extract_digits.py` manages the image segmentation and classification.  We assume that the digits for the problem are written in blue marker; there are no other instance of blue colors in the visual frame. We first threshold the image to extract these blue lines and then fit bounding boxes for each digit. Then, we clip the image to each bounding box, convert the clipping to grayscale, and then classify the digit as 0-9 using a convolutional neural network trained on MNIST (`DigitClassifier.predict()` in `extract_digits.py`). Once we have the classified digits, we group them into two horizontal rows. This grouping works by splitting along the largest vertical gap amongst digits (`group_digits()` in `extract_digits.py`). We find that the camera is low resolution and a bit inconsistent. To mitigate transient issues, we take multiple trials of the image segmentation and use the majority digit classifications. Once we have the final digit results, we perform the addition (`image_callback()` in `arm.py`) to get the digits to draw.

### Inverse Kinematics
The inverse kinematics component controls the arm's movements when drawing the results. The first portion of this process is to find *where* to draw the digits. We make this determination using the location of the problem digits in the image (`get_answer_locations()` in `utils.py`). This process produces the locations to draw in pixels. We convert these pixel locations to distances in meters using calibrated measurements of the size of the visible frame (`image_point_to_draw_digit()` in `utils.py`). From this point, we have the locations in meters of where to draw the digits, and we manually create functions to handle the drawing of each digit (`draw_[digit]` in `arm.py`). We move the arm to the desired locations using inverse kinematics by treating the arm as a 3-joint entity; we always set the arm's joint angle 3 to `0`. Consider moving to a point `(horizontal, vertical)` on the board. Then, `theta0`, the angle for joint `0`, is `theta0 = arctan2(horizontal, length2 + length3 + pen_length)` where `length2` is the length between joints `1` and `2` and `length3` is the length between joints `2`, and `3`. We want to arm to touch the board at the `vertical` position along this `theta0`. Viewed from a horizontal cross-section, this problem is analgous to the 2-joint arm from lecture where `x = horizontal / sin(theta0)` and `y = vertical`. We can thus use the corresponding inverse kinematics to solve for `theta1` and `theta2`. The function `set_arm_position_vertical()` in `arm.py` handles this inverse kinematics logic.

## ROS Node Diagram

## Execution

## Challenges
(1) Originally tried to extract digits using grayscale and assuming it was the only thing drawn on the board, but lighting/glare on board makes this difficult
(2) Joint 2 is set slightly ahead of Joint 1. This throws distances and angles off in the inverse kinematics calculations.

## Future Work
Add more problems types (needs higher set camera for more usuable space on board), handle arbitrary colors on board (better resolution camera), and have turtlebot drive up to the board.

## Takeaways

