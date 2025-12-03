from __future__ import annotations

from geometry_msgs.msg import Quaternion, Point, Vector3, Pose
from interactive_markers import InteractiveMarkerServer, MenuHandler
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
    InteractiveMarkerFeedback,
)
import numpy as np
import csv
from scipy.interpolate import make_interp_spline, CubicSpline
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from collections import namedtuple
import rclpy
import time

MarkerInfo = namedtuple("MarkerInfo", ["x", "y", "yaw"])
PointInfo = namedtuple("PointInfo", ["x", "y", "speed"])

class InteractiveMarkersDemoNode(Node):
    def __init__(self) -> None:
        super().__init__("interactive_markers_demo_node")

        self.declare_parameter("use_file", True)
        self.declare_parameter("control_csv", "control_markers.csv")
        self.declare_parameter("point_csv", "point_markers.csv")
        self.declare_parameter("inter_points", 200)
        self.declare_parameter("speed_min", 2.5)
        self.declare_parameter("speed_max", 4.0)

        self._server = InteractiveMarkerServer(self, "server_name")
        self._marker_num = 0
        self._marker_name_to_xy = {}
        self._marker_names = []
        self._marker_infos = []
        self._data_path = "/home/quantum_drift/ws/src/markers_raceline/data/"

        self.speed_min = self.get_parameter("speed_min").value
        self.speed_max = self.get_parameter("speed_max").value

        self._control_csv_name = self.get_parameter("control_csv").value
        self._points_csv_name = self.get_parameter("point_csv").value
        self._points = []
        self.inter_points = self.get_parameter("inter_points").value

        # TODO: param for deciding to load a file, or start with one
        # marker somewhere

        if self.get_parameter("use_file").value:
            with open(self._data_path + self._control_csv_name) as file:
                for x, y, yaw in csv.reader(file):
                    self.create_marker(float(x), float(y), yaw=float(yaw))
        else:
            self.create_marker(0, 0, 0)
            self.create_marker(1, 0, 0)

        self.line_viz_pub = self.create_publisher(Marker, "/viz_line", 10)
        self.cubic_interpolate()
        self.get_logger().info(f"Initialized {self._marker_num}.")

    def yaw_to_speed(self, yaw):
        # assume yaw in [-pi, pi]
        p = (yaw + np.pi) / (2*np.pi)
        return self.speed_min + p * (self.speed_max - self.speed_min)

    def speed_to_prop(self, speed):
        return (speed - self.speed_min) / (self.speed_max - self.speed_min)

    def create_marker(self, x=0.0, y=0.0, z=0.0, yaw=-np.pi, index=None):
        if index is None:
            index = len(self._marker_names)
        
        # 1. Create a (regular) Marker
        marker = Marker()
        marker.type = Marker.CYLINDER
        marker.scale = Vector3(x=0.4, y=0.4, z=0.1)
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)

        rot_marker = Marker()
        rot_marker.type = Marker.CYLINDER
        rot_marker.scale = Vector3(x=0.6, y=0.6, z=0.05)
        rot_marker.color = ColorRGBA(r=0.1, g=0.1, b=1.0, a=0.5)

        # 2. Create an InteractiveMarkerControl
        move_control = InteractiveMarkerControl()
        move_control.always_visible = True
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        move_control.orientation = Quaternion(w=1.0, x=0.0, y=1.0, z=0.0)
        move_control.markers = [marker]

        rot_control = InteractiveMarkerControl()
        rot_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        rot_control.orientation = Quaternion(w=1.0, x=0.0, y=1.0, z=0.0)
        rot_control.markers = [rot_marker]

        # 3. Create an InteractiveMarker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position = Point(x=float(x), y=float(y), z=float(z))

        q = Rotation.from_euler('z', yaw).as_quat()
        int_marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        int_marker.name = f"m{self._marker_num}"  # give a unique name in the server
        self._marker_num += 1
        # self._marker_name_to_xy[int_marker.name] = (x, y)
        self._marker_names.insert(index, int_marker.name)
        self._marker_infos.insert(index, MarkerInfo(x, y, yaw=yaw))

        int_marker.controls = [move_control, rot_control]  # put the control

        # 4. Add it to the server with a callback
        self._server.insert(int_marker, feedback_callback=self._my_int_marker_feedback_callback)
        self._server.applyChanges()

        # Part 2: Creating a menu handler
        # 1. Create menu handler object
        menu_handler = MenuHandler()

        # 2. Add an option with a callback
        menu_handler.insert("create node", callback=self._menu_create_feedback_callback)
        menu_handler.insert("delete node", callback=self._menu_delete_feedback_callback)
        menu_handler.insert("save nodes", callback=self._menu_save_feedback_callback)
        menu_handler.insert("more points", callback=self._menu_more_feedback_callback)
        menu_handler.insert("fewer points", callback=self._menu_fewer_feedback_callback)

        # 3. Add it to the interactive marker in the server
        menu_handler.apply(self._server, int_marker.name)
        self._server.applyChanges()  # update server


    def _my_int_marker_feedback_callback(
        self, feedback: InteractiveMarkerFeedback
    ) -> None:
        """Prints the marker's current position."""
        marker_name = feedback.marker_name
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            x = feedback.pose.position.x
            y = feedback.pose.position.y
            i = self._marker_names.index(marker_name)

            q = feedback.pose.orientation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            yaw = rot.as_euler('zxy')[0]

            self._marker_infos[i] = MarkerInfo(x, y, yaw=yaw)
            
            self.get_logger().info(
                f"Marker {marker_name!r} {x=:.2f}, {y=:.2f}, {yaw=:.2f}, speed={self.yaw_to_speed(yaw):.2f}"
            )
            self.cubic_interpolate()

    def _menu_create_feedback_callback(self, feedback: InteractiveMarkerFeedback) -> None:
        x = feedback.pose.position.x
        y = feedback.pose.position.y
        marker_name = feedback.marker_name
        index = self._marker_names.index(marker_name)
        self.create_marker(x+1, y, index=index+1)
        self.cubic_interpolate()
        # self.get_logger().info(f"Marker {marker_name!r} says hi!")

    def _menu_delete_feedback_callback(self, feedback: InteractiveMarkerFeedback) -> None:
        marker_name = feedback.marker_name
        index = self._marker_names.index(marker_name)
        self._server.erase(marker_name)
        del self._marker_names[index]
        del self._marker_infos[index]
        self.cubic_interpolate()
        # self.get_logger().info(f"Marker {marker_name!r} says hi!")

    def _menu_save_feedback_callback(self, feedback: InteractiveMarkerFeedback) -> None:
        self.get_logger().info("saving nodes")
        self._save_nodes_to_csv()

    def _save_nodes_to_csv(self):
        with open(self._data_path + self._control_csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self._marker_infos)

        with open(self._data_path + self._points_csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self._points)

    def _menu_more_feedback_callback(self, feedback: InteractiveMarkerFeedback) -> None:
        self.inter_points += 10
        self.cubic_interpolate()

        
    def _menu_fewer_feedback_callback(self, feedback: InteractiveMarkerFeedback) -> None:
        self.inter_points -= 10
        if self.inter_points < 10:
            self.inter_points = 0
        self.cubic_interpolate()


    def cubic_interpolate(self):
        markers = self._marker_infos + self._marker_infos[0:1]
        pos = [[m.x, m.y] for m in markers]
        # pos.append(pos[0])
        pos = np.array(pos)

        dist = np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1))
        t = np.concatenate(([0], np.cumsum(dist)))
        pos_x = pos[:, 0]
        pos_y = pos[:, 1]
        spline_x = CubicSpline(t, pos_x, bc_type='periodic')
        spline_y = CubicSpline(t, pos_y, bc_type='periodic')
        # spline_x = make_interp_spline(t, pos_x, k=3, bc_type='periodic')
        # spline_y = make_interp_spline(t, pos_y, k=3, bc_type='periodic')

        speeds = [self.yaw_to_speed(m.yaw) for m in markers]
        spline_speed = make_interp_spline(t, speeds, k=1)

        t_spaced = np.linspace(t[0], t[-1], self.inter_points)
        x_smooth = spline_x(t_spaced)
        y_smooth = spline_y(t_spaced)
        speed_smooth = spline_speed(t_spaced)
        self._points = [
            PointInfo(x=x, y=y, speed=speed)
            for x, y, speed in zip(x_smooth, y_smooth, speed_smooth)
        ]

        # print([m.yaw for m in markers])
        # print(speeds)
        # print(speed_smooth)

        # self._points = list(zip(x_smooth, y_smooth))
        self.viz_markers()
        
    def viz_markers(self):
        marker = Marker()
        marker.header.frame_id =  "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ""
        marker.id = 0  # adding another marker with id=0 will *replace* the marker
        marker.type = Marker.POINTS
        marker.action = Marker.ADD  # same as Marker.MODIFY

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.01

        # marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        points = []
        colors = []
        for p in self._points:
            points.append(Point(x=p.x, y=p.y, z=0.0))

            g = self.speed_to_prop(p.speed)
            r = 1.0 - g
            c = ColorRGBA(r=r, g=g, b=0.0, a=1.0)
            colors.append(c)

        marker.points = points
        marker.colors = colors

        self.line_viz_pub.publish(marker)
        
def main(args=None) -> None:
    rclpy.init(args=args)
    node = InteractiveMarkersDemoNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
