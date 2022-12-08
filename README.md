# Autonomous-Robot-Math-Solver
Robot autonomously analyzes a math problem on a whiteboard and physically writes out the answer using long addition and long multiplication. Program utilizes computer vision to analyze whiteboard and implements an inverse kinematics algorithm to move the robot arm and draw numbers.

## Team Members
Jay Sivadas, Tejas Kannan, Logan Sherwin, Justin Jones

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
The inverse kinematics component controls the arm's movements when drawing. The first portion of this process is to find *where* to draw the digits. We make this determination using the location of the problem in the image (`get_answer_locations()` in `utils.py`). This process produces the locations to draw in pixels. We convert these pixel locations to distances in meters using hand-calibrated measurements of the visible frame (`image_point_to_draw_digit()` in `utils.py`). We then have the positions in meters of where to draw, and we manually create functions to handle the drawing of each digit (`draw_[digit]` in `arm.py`). We move the arm to these desired locations using inverse kinematics by treating the arm as a 3-joint entity; we always set the arm's joint angle `3` to `0`. Consider moving to a point `(horizontal, vertical)` on the board. Then, `theta0`, the angle for joint `0`, is `theta0 = arctan2(horizontal, length2 + length3 + pen_length)` where `length2` is the length between joints `1` and `2` and `length3` is the length between joints `2`, and `3`. We want to arm to touch the board at the `vertical` position along this `theta0`. Viewed from a horizontal cross-section, this problem is a 2-joint arm where the horizontal distance is `horizontal / sin(theta0)` and vertical distance is `arm_height + vertical`. We use the law of cosines to solve for `theta1` and `theta2`. [This document](calculations/kinematics) contains the full inverse kinematics calculations, and the function `set_arm_position_vertical()` in `arm.py` implements this logic. We made small adjustments to correct for imprecision with the physical Turtlebot. The Turtlebot uses these kinematics to draw the answer and carry digits from least to most significant (`draw_answer_digit()` in `arm.py`).

## ROS Node Diagram
<img width="925" alt="image" src="https://user-images.githubusercontent.com/77175987/206315495-83b3d8ae-6c79-499f-a963-27b4083669b1.png">


## Execution
Note: The following is required for the program to run properly:
- All digits should be written in dark blue ink with other parts in another color. The addition sign should be small.
- Digits should be spaced apart for proper segmentation.
- The front of the robot should be facing directly to the board at a distance of 21cm.

### Process
1. Terminal 1: ```roscore```
2. Terminal 2: ```ssh``` into robot and run ```bringup```
3. Terminal 3: ```ssh``` into robot and run ```bringup_cam```
4. Terminal 4: ```rosrun image_transport republish compressed in:=raspicam_node/image raw out:=camera/rgb/image_raw```
5. Terminal 5: ```roslaunch turtlebot3_manipulation_bringup turtlebot3_manipulation_bringup.launch```
6. Terminal 6: ```roslaunch turtlebot3_manipulation_moveit_config move_group.launch```
7. Terminal 7: ```roslaunch math_solver start.launch```

## Challenges
We initially ran into a few challenges in the actual implementation. In relation to practical implementation of inverse kinematics algorithms, we learned the hard way that precision in measurements is extremely important. We initially hand-measured the lengths of the arms on the robot to use for our inverse kinematics algorithms but this led to innacuracies which caused the robot to not move the arm in the precise direction that we wanted. One major issue was that Joint 2 was set slightly ahead of Joint 1 which caused major issues in the distance and angular calculations of our inverse kinematics.  It wasn't until we got the exact measurements of the arm documentation that we realized this issue. To solve this problem, we set the length of joint 1 to the hypotenuse of the triangle created by this offset, and we explicitly accounted for the angle offset in the kinematics calculations. We also initially tried to use grayscale for digit extraction, but lighting/glare on the whiteboard made this procedure too sensitive. We instead assumed that the digits were written in blue ink and used thresholding to extract the blue lines. Alone, this thresholding led to poor classification due to the camera's low resolution. Thus, after getting the digit bounding boxes, we used adaptive grayscale thresholding within each digit bounding box to get more precise images. This method led to better digit classification.

## Future Work
As the robot currently operates, it can only solve summation questions on the whiteboard. In the future, we hope to add functionality for subtraction, mutiplication, and long-division. Most of the processes would be similar (identifying digits, calculating starting drawing location, and drawing numbers) with slight alterations to the control flow. The long-division would be the most difficult, as its process is completely diferent to addition, subtraction, etc... We would also have to put in more work to define a classifier that can identify the function (addition, subtraction, multiplication, etc...) and not just the individual numbers. We also would want to extend current functionality such as having a higher set camera or view of the board to have better board space usages and larger math problems, handling arbitrary colors and not just blue, and having the turtlebot drive to the board to start drawing. 

## Takeaways
1. The importance of teamwork in developing ideas from the ground up.
- The project we implemented had multiple moving parts ranging from digit recognition, to calculations, to actual arm programming. Some parts were independent and some parts directly relied on eachother to operate. 
- In planning the development of the actual development, our teamwork caused the design and outline to go as smoothly as possible. We were able to discuss what part we needed and brainstorm together how to tackle challenges.
- The final robot result was a product of us starting from the beginning with a clear vision of what we wanted due to the initial work the entire group put in together to provide that clarity.
2. The difficulties of inverse kinematics in achieving complex functional movements with pinpoint accuracy.
- Before this project, we didn't really understand the actual difficulty it requires to calculate the outputs in inverse kinematics. It seemed like pure math, and with a few degrees of change, it seemed relatively straightforward. 
- Actually implementing the algorithms required us to undergoe trial and error multiple times to sensitize it to the actual real world environment. At even just 3 angles, there were so many moving parts we had to account for, and possible frictions in the environment that could impact results.
- The difficulty it must take to implement inverse kinematics algorithms with even more joints, especially in situations like robot surgery programming that requires millimeters of precisions, seems much more daunting after this project.

## Demo
The GIFs below show examples of the Turtlebot solving addition problems on a whiteboard. These same videos are in [this Google Drive folder](https://drive.google.com/drive/folders/1yF00Exp_nhJEt3AJx4cHzMSwA0S5E9J6?usp=share_link).

![290 + 82](gifs/add0.gif)
