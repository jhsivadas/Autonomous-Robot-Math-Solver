#! /usr/bin/python3

import rospy

# import the moveit_commander, which allows us to control the arms
import moveit_commander
import numpy as np

# import the custom message
from math_solver.msg import Arm


class Robot(object):
    def __init__(self):

        # initialize this node
        rospy.init_node("turtlebot3_dance")

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # Arm status subscriber
        rospy.Subscriber("/arm_status", Arm, self.arm_dir_received)
        rospy.sleep(0.5)

        # Reset arm position
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        print("ready")

    def arm_dir_received(self, data: Arm):
        # select location based on data direction
        arm_joint_0_goal = data.direction0
        arm_joint_1_goal = data.direction1
        arm_joint_2_goal = data.direction2
        arm_joint_3_goal = data.direction3
        print(
            f"0: {np.degrees(arm_joint_0_goal)}, 1: {np.degrees(arm_joint_1_goal)}, 2: {np.degrees(arm_joint_2_goal)}, 3: {np.degrees(arm_joint_3_goal)}"
        )

        gripper_joint_open = [0, 0]

        print("moving")
        print(arm_joint_1_goal)
        # wait=True ensures that the movement is synchronous
        self.move_group_arm.go(
            [arm_joint_0_goal, arm_joint_1_goal, arm_joint_2_goal, arm_joint_3_goal],
            wait=True,
        )
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()

        self.move_group_gripper.go(gripper_joint_open, wait=True)
        self.move_group_gripper.stop()

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    robot = Robot()
    robot.run()
