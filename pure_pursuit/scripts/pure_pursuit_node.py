#!/usr/bin/env python3
import csv, math, os
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from collections import namedtuple
import numpy as np

PointInfo = namedtuple("PointInfo", ["x", "y", "speed"])

def load_path(csv_path: str) -> List[Tuple[float,float]]:
    pts=[]
    with open(csv_path,'r') as f:
        for row in csv.reader(f):
            if len(row) < 2: continue
            try:
                pt = PointInfo(x=float(row[0]), y=float(row[1]), speed=float(row[2]))
                pts.append(pt)
            except: pass
    if len(pts)==0:
        raise RuntimeError(f'No points loaded from {csv_path}')
    return pts

def wrap_angle(a):
    while a> math.pi:  a-=2*math.pi
    while a<-math.pi: a+=2*math.pi
    return a

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # # sim 
        # self.csv_in   = self.declare_parameter('csv_in','/sim_ws/src/markers_raceline/data/point_markers.csv').get_parameter_value().string_value
        # real car
        self.csv_in   = self.declare_parameter('csv_in',f'/home/quantum_drift/ws/src/markers_raceline/data/point_markers.csv').get_parameter_value().string_value

        self.frame_id = self.declare_parameter('frame_id','map').get_parameter_value().string_value
        self.base_link= self.declare_parameter('base_link','ego_racecar/base_link').get_parameter_value().string_value
        self.L        = float(self.declare_parameter('lookahead',1.9).get_parameter_value().double_value)
        self.WB       = float(self.declare_parameter('wheelbase',0.33).get_parameter_value().double_value)
        self.speed    = float(self.declare_parameter('speed',3.0).get_parameter_value().double_value)
        self.cmd_topic= self.declare_parameter('cmd_topic','/drive').get_parameter_value().string_value
        self.odom_topic= self.declare_parameter('odom_topic','pf/pose/odom').get_parameter_value().string_value
        # self.odom_topic= self.declare_parameter('odom_topic','ego_racecar/odom').get_parameter_value().string_value

        self.do_reactive_speed = self.declare_parameter('do_reactive_speed', True).get_parameter_value().bool_value
        self.rspeed_min = self.declare_parameter('rspeed_min', 1.2).get_parameter_value().double_value
        self.rspeed_max = self.declare_parameter('rspeed_max', 5.5).get_parameter_value().double_value


        self.path_pts = load_path(self.csv_in)
        self.get_logger().info(f'Loaded {len(self.path_pts)} path points from {self.csv_in}')

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        self.x = 0
        self.y = 0
        self.yaw = 0
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.goal_viz_pub = self.create_publisher(Marker, "/viz_goal", 10)

        # self.timer = self.create_timer(1.0/30.0, self.control_step)  # 30 Hz

    def odom_cb(self, msg : Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        self.control_step()

    def control_step(self):
        x, y, yaw = self.x, self.y, self.yaw
        # print(f"{x=}, {y=}")
        # find goal point ~L ahead measured along straight-line distance from current pose
        gx,gy,speed = self.select_goal(x,y,yaw)
        if gx is None:
            self.get_logger().warn('No goal found'); return

        self.viz_goal(gx, gy)

        # transform goal into vehicle (base_link) frame
        dx = gx - x; dy = gy - y
        # rotate by -yaw
        lx =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
        ly =  math.sin(-yaw)*dx + math.cos(-yaw)*dy

        # pure pursuit curvature & steering
        # kappa = 2*y_ld / L^2 ; delta = atan(WB * kappa)
        if self.L < 1e-3: return
        kappa = 2.0 * ly / (self.L**2)
        delta = math.atan(self.WB * kappa)
        # delta = kappa 
        
        # if self.do_reactive_speed:
        #     base = rspeed_min + speed_gain * max(0.0, goal_dist)
        # # base = max_speed
        # curve_factor = 1.0 / (1.0 + curv_slow * abs(delta))
        # speed = float(np.clip(base * curve_factor, min_speed, max_speed))


        # base = min_speed + speed_gain * max(0.0, goal_dist)
        # # base = max_speed
        # curve_factor = 1.0 / (1.0 + curv_slow * abs(delta))
        # speed = float(np.clip(base * curve_factor, min_speed, max_speed))

        if np.isnan(delta) or np.isnan(speed):
            print("nan")
            return

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        print(delta, speed)
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def select_goal(self, x, y, yaw) -> Tuple[float,float]:
        path_pts = np.array(self.path_pts)
        path_xy = path_pts[:,:2]
        # get angle to all path points (car frame)
        path_angles = np.arctan2(path_xy[:,1]-y, path_xy[:,0]-x) - yaw
        # wrap angles
        path_angles = np.abs((path_angles + np.pi) % (2*np.pi) - np.pi)
        # distance from car to path points
        dist = np.sqrt(np.sum((path_xy - [x, y])**2, axis=1))
        # some constants
        Lmin = 0.5*self.L; Lmax = 1.5*self.L
        max_angle = 2.0 # ~115deg
        # only consider points that meet all requirements
        mask = (path_angles < max_angle) & (dist > Lmin) & (dist < Lmax)
        # if no valid points, return
        if np.sum(mask) == 0: return None, None, None
        # find point closest to lookahead
        i = np.argmin(np.abs(dist[mask] - self.L))
        # return point coords (map frame)
        gx, gy = path_xy[mask][i]
        speed = path_pts[mask][i,2]
        return gx, gy, speed

    def viz_goal(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ""
        marker.id = 0 
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0 

        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.01

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        self.goal_viz_pub.publish(marker)
        

    

def main(args=None):
    print("Hello")
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
