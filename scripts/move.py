#! /usr/bin/python3

import rospy
# import the moveit_commander, which allows us to control the arms
import moveit_commander
import math
import numpy as np
# import the custom message
from math_solver.msg import Traffic

class Robot(object):

    def __init__(self):

        # initialize this node
        rospy.init_node('turtlebot3_dance')

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # Traffic status subscriber
        rospy.Subscriber("/traffic_status", Traffic, self.traffic_dir_received)
        rospy.sleep(0.5)

        # Reset arm position
        self.move_group_arm.go([0,0,0,0], wait=True)
        print("ready")

    def traffic_dir_received(self, data: Traffic):
        # array of arm joint locations for joint 0
        arm_joint_0 = [math.pi/2, 0, -1 * math.pi/2]

        # select location based on data direction
        # arm_joint_0_goal = arm_joint_0[data.direction]
        arm_joint_0_goal = data.direction0
        arm_joint_1_goal = data.direction1
        arm_joint_2_goal = data.direction2
        arm_joint_3_goal = data.direction3

        gripper_joint_close = [-0.01, -0.01]
        gripper_joint_open = [0, 0]

        # self.move_group_gripper.go(gripper_joint_close)
        # self.move_group_gripper.stop()
        print("moving")
        # wait=True ensures that the movement is synchronous
        self.move_group_arm.go([arm_joint_0_goal, arm_joint_1_goal, arm_joint_2_goal, arm_joint_3_goal], wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()

        # self.move_group_arm.go([arm_joint_0_goal, 0, 0, -1 * math.pi/4], wait=True)
        # self.move_group_arm.stop()

        self.move_group_gripper.go(gripper_joint_open, wait=True)
        self.move_group_gripper.stop()

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    robot = Robot()
    robot.run()