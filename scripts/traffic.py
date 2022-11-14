#! /usr/bin/python3

import rospy
from math_solver.msg import Traffic
import numpy as np

class TrafficNode(object):
    def __init__(self):
        # Set up traffic status publisher
        self.traffic_status_pub = rospy.Publisher("/traffic_status", Traffic, queue_size=10)

        # Counter to loop publishing direction with
        self.direction_counter = 0
        rospy.sleep(1)

    def run(self):
        # Once every 10 seconds
        sleep_time = 5
        while (not rospy.is_shutdown()):
            trafficMsg = Traffic()
            # trafficMsg.direction = self.direction_counter % 3
            # self.direction_counter += 1
            trafficMsg.direction0 = 0
            trafficMsg.direction1 = 0
            trafficMsg.direction2 = 0
            trafficMsg.direction3 = 0
            self.traffic_status_pub.publish(trafficMsg)
            rospy.sleep(5)
            # horizontal move to the left, 8in
            # joint0 -> joint1 = 6in, joint1 -> marker tip = 16in
            # trafficMsg.direction0 = np.arctan(6/16)
            # trafficMsg.direction1 = np.arcsin((np.sqrt(8**2+16**2)-16)/(6))
            # trafficMsg.direction2 = -trafficMsg.direction1
            # trafficMsg.direction3 = 0

            # vertical move up, 4in
            # joint3 -> marker tip = 9in
            trafficMsg.direction0 = 0
            trafficMsg.direction1 = np.arcsin((np.sqrt(4**2+16**2)-16)/(6))
            trafficMsg.direction2 = np.arctan(4/16)
            trafficMsg.direction3 = -trafficMsg.direction1

            self.traffic_status_pub.publish(trafficMsg)
            rospy.sleep(5)

if __name__ == '__main__':
    rospy.init_node('traffic_controller')
    traffic_node = TrafficNode()
    traffic_node.run()