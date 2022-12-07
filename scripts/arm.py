#! /usr/bin/python3

import rospy
import math
import numpy as np
import os.path
import cv2, cv_bridge

from collections import namedtuple
from enum import Enum, auto
from math_solver.msg import Arm
from sensor_msgs.msg import Image
from typing import Tuple, List

from constants import (
    JOINT1_LENGTH,
    JOINT2_LENGTH,
    JOINT3_LENGTH,
    PEN_LENGTH,
    BOX_LENGTH,
)
from utils import (
    inches_to_meters,
    get_answer_locations,
    add_numbers,
    image_point_to_draw_digit,
    DrawDigit,
    consolidate_digits,
)
from extract_digits import extract_digits, DigitClassifier, Digit


Point = namedtuple("Point", ["x", "y", "z"])
VERTICAL_ADJUSTMENT = -inches_to_meters(3.5)
HORIZ_ADJUSTMENT = 0.05
PULLOFF_DIST = 0.07
VERTICAL_NUM_MULTIPLIER = 1.75
NUMBER_SIZE = 0.75


class ControlMode(Enum):
    EXTRACT_DIGITS = auto()
    DRAW = auto()
    COMPLETED = auto()


class ArmNode(object):
    def __init__(self):
        rospy.init_node("arm_node")

        self.l1 = JOINT1_LENGTH  # base to joint2
        self.l2 = JOINT2_LENGTH  # joint2 to joint3
        self.pen = PEN_LENGTH  # center of gripper to pen tip
        self.l3 = JOINT3_LENGTH + self.pen  # joint3 to end of gripper
        self.box = BOX_LENGTH
        self.phi = np.arcsin(self.box / self.l1)
        self.arm_height = np.sqrt(
            np.square(self.l1) - np.square(self.box)
        )  # The vertical height of the arm at (0, 0)

        # Set up status publisher
        self.arm_status_pub = rospy.Publisher("/arm_status", Arm, queue_size=10)

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # initalize the debugging window
        cv2.namedWindow("window", 1)

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber(
            "camera/rgb/image_raw", Image, self.image_callback
        )

        # Loads stored digit classifier
        folder = os.path.dirname(__file__)
        self.digit_classifier = DigitClassifier(os.path.join(folder, "mnist.h5"))

        # Counter to loop publishing direction with
        self.direction_counter = 0

        # Initializes current position of arm
        self.last_msg = None
        self.current_pos = Point(
            self.box + self.l2 + self.l3, self.arm_height + VERTICAL_ADJUSTMENT, 0.0
        )
        self.control_mode = ControlMode.EXTRACT_DIGITS

        self.draw_width = inches_to_meters(NUMBER_SIZE)
        self.draw_digits: List[DrawDigit] = []

        self.num_digit_trials = 5
        self.digit_trial_counter = 0
        self.top_digit_trials: List[List[Digit]] = []
        self.bottom_digit_trials: List[List[Digit]] = []

        # Sets up dispatch table to draw each digit
        self.dispatch = {
            0: self.draw_zero,
            1: self.draw_one,
            2: self.draw_two,
            3: self.draw_three,
            4: self.draw_four,
            5: self.draw_five,
            6: self.draw_six,
            7: self.draw_seven,
            8: self.draw_eight,
            9: self.draw_nine,
        }

        rospy.sleep(1)

    def image_callback(self, msg):
        if self.control_mode != ControlMode.EXTRACT_DIGITS:
            return

        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        top_number_digits, bottom_number_digits = extract_digits(
            image, self.digit_classifier
        )

        self.top_digit_trials.append(top_number_digits)
        self.bottom_digit_trials.append(bottom_number_digits)

        self.digit_trial_counter += 1

        # Stop if we hae not taken enough trials yet
        if self.digit_trial_counter < self.num_digit_trials:
            return

        # Otherwise, process the digits using the majority counts over all of the trials
        consolidated_top_digits: List[Digit] = consolidate_digits(
            self.top_digit_trials
        )  # Top number digits in order of most significant to least significant
        consolidated_bottom_digits: List[Digit] = consolidate_digits(
            self.bottom_digit_trials
        )  # Bottom number digits in order of most significant to least significant

        # Add the two values
        add_digits, add_carry = add_numbers(
            consolidated_top_digits, consolidated_bottom_digits
        )

        # Get the locations (in pixels) where we should start the result drawings
        draw_locations, carry_locations = get_answer_locations(
            consolidated_top_digits,
            consolidated_bottom_digits,
            num_digits=len(add_digits),
            draw_width=self.draw_width,
        )

        # Iterates through each digit to be drawn and its location
        self.digits_to_draw: List[DrawDigit] = []
        for loc, digit in zip(draw_locations, add_digits):
            draw_digit = image_point_to_draw_digit(
                image_point=loc,
                digit_value=digit,
                image_width=image.shape[1],
                vertical_starting_point=VERTICAL_ADJUSTMENT,
            )

            # Adjusts location of the 1 digit so it draws more centrally instead
            # of along the left edge of its bounding box
            if digit == 1:
                draw_digit = DrawDigit(
                    horizontal=draw_digit.horizontal - float(self.draw_width / 2.0),
                    vertical=draw_digit.vertical,
                    digit=draw_digit.digit,
                )

            self.digits_to_draw.append(draw_digit)

        self.carry_digits_to_draw: List[DrawDigit] = []
        if len(add_carry) == len(carry_locations) + 1:
            add_carry = add_carry[:-1]

        print(f"Carry locations: {carry_locations}")
        print(f"Add Carry: {add_carry}")

        for loc, digit in zip(carry_locations, add_carry):
            draw_carry_digit = image_point_to_draw_digit(
                image_point=loc,
                digit_value=digit,
                image_width=image.shape[1],
                vertical_starting_point=VERTICAL_ADJUSTMENT,
            )

            # When drawing a 1, start in the middle of the box (as opposed to
            # the left) because there is no horizontal component
            if digit == 1:
                draw_carry_digit = DrawDigit(
                    horizontal=draw_carry_digit.horizontal
                    - float(self.draw_width / 3.0),
                    vertical=draw_carry_digit.vertical,
                    digit=draw_carry_digit.digit,
                )

            self.carry_digits_to_draw.append(draw_carry_digit)

        for loc in draw_locations:
            cv2.circle(
                image, (loc.horizontal_pixels, loc.vertical_pixels), 3, (0, 0, 255), 3
            )

        print(
            "Top Number: {}".format(
                "".join(map(str, map(lambda d: d.value, consolidated_top_digits)))
            )
        )
        print(
            "Bottom Number: {}".format(
                "".join(map(str, map(lambda d: d.value, consolidated_bottom_digits)))
            )
        )
        print("Sum: {}".format("".join(map(str, reversed(add_digits)))))

        # cv2.imshow("window", image)
        # cv2.waitKey(3)

        self.control_mode = ControlMode.DRAW

    def change_dist(self, x: float):
        self.current_pos = Point(
            self.box + self.l2 + self.l3 - x, self.current_pos.y, self.current_pos.z
        )
        last_msg = self.set_arm_position_vertical(
            self.current_pos.x, self.current_pos.y
        )
        theta = math.atan2(self.current_pos.z, self.box + self.l2 + self.l3)
        last_msg.direction0 = theta
        return last_msg

    def get_reset_msg(self):
        curr_dist_from_wall = self.box + self.l2 + self.l3 - PULLOFF_DIST
        curr_height = self.arm_height + VERTICAL_ADJUSTMENT

        self.current_pos = Point(curr_dist_from_wall, curr_height, 0.0)

        self.last_msg = self.set_arm_position_vertical(curr_dist_from_wall, curr_height)
        print("Reset{}".format(self.current_pos))

        return self.last_msg

    def set_arm_position_vertical(self, target_x: float, target_y: float) -> Arm:
        """
        Sets the arm position to (target_x, target_y) where `x` is the robot's distance from wall (along the angle of theta_0)
        and `y` is the vertical height (where 0 is the base of the robot's arm).

        The `origin` of this system is (target_x = self.box + self.l2 + self.l3, target_y = self.arm_height). All values in the (x, y)
        place should be displaced off of this origin.
        """
        arm1_length = self.l1
        arm2_length = (
            self.l2 + self.l3
        )  #  We treat the second two joints as a single arm (never change theta_3)

        d_squared = target_x**2 + target_y**2
        cos_alpha = (arm1_length**2 + arm2_length**2 - d_squared) / (
            2 * arm1_length * arm2_length
        )
        alpha = np.arccos(cos_alpha)

        # TODO: Maybe break ties based on the better angle with the board (or
        # use Joint 3 to remove need entirely)
        q2 = math.pi - alpha
        theta2 = (math.pi / 2.0) - alpha + self.phi

        beta = np.arctan2(
            arm2_length * np.sin(q2), arm1_length + (arm2_length * np.cos(q2))
        )
        q1 = np.arctan2(target_y, target_x) + beta
        theta1 = (math.pi / 2.0) - self.phi - q1

        arm_msg = Arm()
        arm_msg.direction0 = 0.0
        arm_msg.direction1 = theta1
        arm_msg.direction2 = theta2 - math.radians(12)
        arm_msg.direction3 = 0.0

        return arm_msg

    def get_horizontal_msg(self, target, right=False):
        if right:
            target *= -1

        origin_dist_from_wall = self.box + self.l2 + self.l3

        theta = math.atan2((target + self.current_pos.z), origin_dist_from_wall)
        new_dist_from_wall = np.sqrt(
            (target + self.current_pos.z) ** 2 + origin_dist_from_wall**2
        )

        arm_msg = self.set_arm_position_vertical(
            target_x=new_dist_from_wall, target_y=self.current_pos.y
        )

        arm_msg.direction0 = theta

        new_pos = Point(
            new_dist_from_wall, self.current_pos.y, target + self.current_pos.z
        )
        self.current_pos = new_pos

        return arm_msg

    def get_vertical_msg(self, target, up=False):
        if up:
            target *= -1

        # Updates arm positions based on target
        curr_dist_from_wall = self.current_pos.x
        new_vertical_position = self.current_pos.y - target

        # Adjusts arm position
        arm_msg = self.set_arm_position_vertical(
            target_x=curr_dist_from_wall, target_y=new_vertical_position
        )

        theta = math.atan2(self.current_pos.z, self.box + self.l2 + self.l3)
        arm_msg.direction0 = theta

        new_pos = Point(self.current_pos.x, new_vertical_position, self.current_pos.z)
        self.current_pos = new_pos

        return arm_msg

    def draw_zero(self, sleep_time, target):
        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_one(self, sleep_time, target):
        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_two(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_three(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_four(self, sleep_time, target):
        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_five(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_six(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_seven(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_eight(self, sleep_time, target):
        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * 1.2, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def draw_nine(self, sleep_time, target):
        right_msg = self.get_horizontal_msg(target, right=True)
        self.arm_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * VERTICAL_NUM_MULTIPLIER, up=False)
        self.arm_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        left_msg = self.get_horizontal_msg(target, right=False)
        self.arm_status_pub.publish(left_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.arm_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

    def get_horizontal_parameters(self, horizontal_dist: float) -> Tuple[float, float]:
        # Move the arm to the start location
        origin_dist_to_wall = self.box + self.l2 + self.l3
        theta0 = np.arctan2(horizontal_dist, origin_dist_to_wall)
        print(f"Theta 0: {theta0}")
        dist_to_wall = np.sqrt(horizontal_dist**2 + origin_dist_to_wall**2) + (
            0.003 * abs(theta0)
        )
        return theta0, dist_to_wall

    def draw_answer_digit(self, digit: DrawDigit, sleep_time: int, is_carry: bool):
        print("Drawing answer digit")
        # Pull arm off the wall
        self.arm_status_pub.publish(self.change_dist(PULLOFF_DIST))
        rospy.sleep(sleep_time)

        # Move the arm to the start location
        theta0, dist_to_wall = self.get_horizontal_parameters(digit.horizontal)

        start_msg = self.set_arm_position_vertical(
            target_x=dist_to_wall - PULLOFF_DIST,
            target_y=self.arm_height + digit.vertical,
        )
        start_msg.direction0 = theta0

        self.arm_status_pub.publish(start_msg)
        rospy.sleep(sleep_time)

        start_msg = self.set_arm_position_vertical(
            target_x=dist_to_wall, target_y=self.arm_height + digit.vertical
        )

        start_msg.direction0 = theta0

        self.arm_status_pub.publish(start_msg)
        rospy.sleep(sleep_time)

        new_pos = Point(
            dist_to_wall, self.arm_height + digit.vertical, digit.horizontal
        )
        self.current_pos = new_pos

        if is_carry:
            num_size = inches_to_meters(NUMBER_SIZE * 0.75)
        else:
            num_size = inches_to_meters(NUMBER_SIZE)
        # Gets the appropriate digit-drawing function and calls it
        self.dispatch[digit.digit](sleep_time, num_size)
        rospy.sleep(sleep_time)

    def run(self):
        sleep_time = 3
        self.digit_trial_counter = 0

        while not rospy.is_shutdown():
            if self.control_mode == ControlMode.DRAW:
                # Reset the arm position
                reset_msg = self.get_reset_msg()
                self.arm_status_pub.publish(reset_msg)
                rospy.sleep(sleep_time)

                for i in range(len(self.digits_to_draw)):
                    if (
                        i < len(self.carry_digits_to_draw)
                        and self.carry_digits_to_draw[i].digit != 0
                    ):
                        self.draw_answer_digit(
                            self.carry_digits_to_draw[i], sleep_time, True
                        )

                    self.draw_answer_digit(self.digits_to_draw[i], sleep_time, False)
                self.arm_status_pub.publish(self.get_reset_msg())
                self.control_mode = ControlMode.COMPLETED
                rospy.signal_shutdown("Completed")


if __name__ == "__main__":
    arm_node = ArmNode()
    arm_node.run()
