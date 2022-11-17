#! /usr/bin/python3

import rospy
from math_solver.msg import Traffic
import numpy as np

class TrafficNode(object):
    def __init__(self):
        self.l1 = 0.130   # base to joint2
        self.l2 = 0.124   # joint2 to joint3
        self.pen = 0.114  # center of gripper to pen tip
        self.l3 = 0.126 + self.pen   # joint3 to end of gripper
        self.box = 0.024
        self.phi = np.arcsin(self.box / self.l1)

        # Set up traffic status publisher
        self.traffic_status_pub = rospy.Publisher("/traffic_status", Traffic, queue_size=10)

        # Counter to loop publishing direction with
        self.direction_counter = 0

        self.last_msg = None

        rospy.sleep(1)


    def get_reset_msg(self):
        self.last_msg = Traffic(0, 0, 0, 0)
        return self.last_msg


    def get_horizontal_msg(self, target, right=False):
        trafficMsg = Traffic()

        if right:
            target *= -1

        delta = np.sqrt((self.l2+self.l3)**2 + target**2) - (self.l2 + self.l3)

        trafficMsg.direction0 = np.arctan(target / (self.l2 + self.l3))
        trafficMsg.direction1 = np.arcsin(delta / self.l1)
        trafficMsg.direction2 = -trafficMsg.direction1
        trafficMsg.direction3 = self.last_msg.direction3

        self.last_msg = trafficMsg

        return trafficMsg


    def get_vertical_msg(self, target, up=False):
        trafficMsg = Traffic()

        if up:
            target *= -1

        delta2 = np.sqrt((self.l2+self.l3)**2 - target**2)
        delta1 = self.box + self.l2 + self.l3 - delta2
        alpha = np.arcsin(target / (self.l2 + self.l3))

        print(f"alpha: {np.degrees(alpha)}, D1: {np.degrees(delta1)}, D2: {np.degrees(delta2)}, Phi: {np.degrees(self.phi)}")
        trafficMsg.direction0 = self.last_msg.direction0
        trafficMsg.direction1 = (np.arcsin(delta1 / self.l1) - self.phi)
        trafficMsg.direction2 = alpha - self.phi - trafficMsg.direction1
        trafficMsg.direction3 = self.last_msg.direction3

        self.last_msg = trafficMsg

        return trafficMsg


    def draw_four(self, sleep_time, target):
        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg(target, right=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target * 2, up=True))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target * 2, up=False))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_horizontal_msg(target, right=True))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target * 2, up=True))
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_reset_msg())
        rospy.sleep(sleep_time)

        self.traffic_status_pub.publish(
            self.get_vertical_msg(target * 2, up=False))
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


    def run(self):
        sleep_time = 3
        target = 3 * .0254   # 3" converted to m
        
        while (not rospy.is_shutdown()):
            self.draw_four(sleep_time, target)

if __name__ == '__main__':
    rospy.init_node('traffic_controller')
    traffic_node = TrafficNode()
    traffic_node.run()