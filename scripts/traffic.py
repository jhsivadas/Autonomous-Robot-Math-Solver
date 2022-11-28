#! /usr/bin/python3

import rospy
import math
import numpy as np

from math_solver.msg import Traffic
from constants import JOINT1_LENGTH, JOINT2_LENGTH, JOINT3_LENGTH
from constants import PEN_LENGTH, BOX_LENGTH
from utils import inches_to_meters
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y', 'z'])
VERTICAL_ADJUSTMENT = 0
HORIZ_ADJUSTMENT = 0.05



class TrafficNode(object):
    def __init__(self):
        self.l1 = JOINT1_LENGTH   # base to joint2
        self.l2 = JOINT2_LENGTH   # joint2 to joint3
        self.pen = PEN_LENGTH  # center of gripper to pen tip
        self.l3 = JOINT3_LENGTH + self.pen   # joint3 to end of gripper
        self.box = BOX_LENGTH
        self.phi = np.arcsin(self.box / self.l1)
        self.arm_height = np.sqrt(np.square(self.l1) - np.square(self.box))  # The vertical height of the arm at (0, 0)

        # Set up traffic status publisher
        self.traffic_status_pub = rospy.Publisher("/traffic_status", Traffic, queue_size=10)

        # Counter to loop publishing direction with
        self.direction_counter = 0

        self.last_msg = None

        self.current_pos = Point(self.box + self.l2 + self.l3, self.arm_height + VERTICAL_ADJUSTMENT, 0.0)

        rospy.sleep(1)

    def change_dist(self, x = 0.0):
        self.current_pos = Point(self.box + self.l2 + self.l3 - x, self.current_pos.y, self.current_pos.z)
        last_msg = self.set_arm_position_vertical(self.current_pos.x, self.current_pos.y)
        theta = math.atan2(self.current_pos.z, self.box + self.l2 + self.l3)
        last_msg.direction0 = theta
        return last_msg


    def get_reset_msg(self):
        #self.last_msg = Traffic(0, 0, 0, 0)
        #return self.last_msg

        curr_dist_from_wall = self.box + self.l2 + self.l3
        curr_height = self.arm_height + VERTICAL_ADJUSTMENT

        self.current_pos = Point(curr_dist_from_wall, curr_height, 0.0)

        self.last_msg = self.set_arm_position_vertical(curr_dist_from_wall, curr_height)
        print("Reset{}".format(self.current_pos))

        return self.last_msg


    def set_arm_position_vertical(self, target_x: float, target_y: float) -> Traffic:
        """
        Sets the arm position to (target_x, target_y) where `x` is the robot's distance from wall (along the angle of theta_0)
        and `y` is the vertical height (where 0 is the base of the robot's arm).

        The `origin` of this system is (target_x = self.box + self.l2 + self.l3, target_y = self.arm_height). All values in the (x, y)
        place should be displaced off of this origin.
        """
        arm1_length = self.l1
        arm2_length = self.l2 + self.l3  #  We treat the second two joints as a single arm (never change theta_3)

        d_squared = target_x**2 + target_y**2
        cos_alpha = (arm1_length**2 + arm2_length**2 - d_squared) / (2 * arm1_length * arm2_length)
        alpha = np.arccos(cos_alpha)

        # TODO: Maybe break ties based on the better angle with the board (or use Joint 3 to remove need entirely)
        q2 = math.pi - alpha
        theta2 = (math.pi / 2.0) - alpha + self.phi

        beta = np.arctan2(arm2_length * np.sin(q2), arm1_length + (arm2_length * np.cos(q2)))
        q1 = np.arctan2(target_y, target_x) + beta
        theta1 = (math.pi / 2.0) - self.phi - q1

        arm_msg = Traffic()
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
        new_dist_from_wall = np.sqrt((target + self.current_pos.z)**2 + origin_dist_from_wall**2) 
        

        # new_dist_from_wall = np.sqrt(target**2 + origin_dist_from_wall**2) + .006
        
        arm_msg = self.set_arm_position_vertical(target_x=new_dist_from_wall,
                                                 target_y=self.current_pos.y)

        arm_msg.direction0 = theta

        new_pos = Point(new_dist_from_wall, self.current_pos.y, target + self.current_pos.z)
        self.current_pos = new_pos
        print("Current Position: {}, theta: {}".format(self.current_pos, theta))

        return arm_msg

        #trafficMsg = Traffic()
        #delta = np.sqrt((self.l2+self.l3)**2 + target**2) - (self.l2 + self.l3)

        #trafficMsg.direction0 = np.arctan(target / (self.l2 + self.l3))
        #trafficMsg.direction1 = np.arcsin(delta / self.l1)
        #trafficMsg.direction2 = -trafficMsg.direction1
        #trafficMsg.direction3 = self.last_msg.direction3

        #self.last_msg = trafficMsg

        #return trafficMsg


    def get_vertical_msg(self, target, up=False):
        if up:
            target *= -1

        curr_dist_from_wall = self.current_pos.x
        new_vertical_position = self.current_pos.y - target

        arm_msg = self.set_arm_position_vertical(target_x=curr_dist_from_wall,
                                                 target_y=new_vertical_position)

        theta = math.atan2(self.current_pos.z, self.box + self.l2 + self.l3)
        arm_msg.direction0 = theta

        new_pos = Point(self.current_pos.x, new_vertical_position, self.current_pos.z)
        self.current_pos = new_pos

        return arm_msg

        #trafficMsg = Traffic()

        #delta2 = np.sqrt((self.l2+self.l3)**2 - target**2)
        #delta1 = self.box + self.l2 + self.l3 - delta2
        #alpha = np.arcsin(target / (self.l2 + self.l3))

        #print(f"alpha: {np.degrees(alpha)}, D1: {np.degrees(delta1)}, D2: {np.degrees(delta2)}, Phi: {np.degrees(self.phi)}")
        #trafficMsg.direction0 = self.last_msg.direction0
        #trafficMsg.direction1 = (np.arcsin(delta1 / self.l1) - self.phi)
        #trafficMsg.direction2 = alpha - self.phi - trafficMsg.direction1
        #trafficMsg.direction3 = self.last_msg.direction3

        #self.last_msg = trafficMsg

        #return trafficMsg


    

    def draw_four(self, sleep_time, target):
        self.traffic_status_pub.publish(self.get_reset_msg())
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target, up=False)
        self.traffic_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        right_msg = self.get_horizontal_msg(target, right=True)
        self.traffic_status_pub.publish(right_msg)
        rospy.sleep(sleep_time)

        up_msg = self.get_vertical_msg(target, up=True)
        self.traffic_status_pub.publish(up_msg)
        rospy.sleep(sleep_time)

        down_msg = self.get_vertical_msg(target * 2, up=False)
        self.traffic_status_pub.publish(down_msg)
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(self.change_dist(.01))
        rospy.sleep(sleep_time)
        self.traffic_status_pub.publish(self.get_reset_msg())
        rospy.sleep(sleep_time)





    def draw_seven(self, sleep_time, target):
        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg(target, right=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target * 2, up=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

    def draw_left(self, sleep_time, target):
        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg(target, right=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg(target, right=False))
        rospy.sleep(sleep_time)

    def draw_up(self, sleep_time, target):
        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg (target, right=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target, up=True))
        rospy.sleep(sleep_time)

      

        




    def run(self):
        sleep_time = 3
        target = inches_to_meters(1)    # 3" converted to m

        while (not rospy.is_shutdown()):
            self.draw_four(sleep_time, target)

if __name__ == '__main__':
    rospy.init_node('traffic_controller')
    traffic_node = TrafficNode()
    traffic_node.run()
