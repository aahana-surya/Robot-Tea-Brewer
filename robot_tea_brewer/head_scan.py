"""ROS 2 node: sweep the Stretch head pan/tilt until the tag localizer sees the tag."""
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint


class HeadScan(Node):
    def __init__(self):
        super().__init__('head_scan')
        self.declare_parameter('action_name', '/stretch_controller/follow_joint_trajectory')
        self.declare_parameter('pan_center', -1.1)  # rad; 0 = straight ahead, negative = right (table side)
        self.declare_parameter('pan_range', 0.5)    # sweep +/- this much
        self.declare_parameter('pan_steps', 5)
        self.declare_parameter('tilts', [-0.8, -1.0])  # rad; more negative = looks further down
        self.declare_parameter('move_time', 2.0)    # s per move
        self.declare_parameter('dwell', 1.5)        # s to let the localizer look

        p = self.get_parameter
        c, r, n = p('pan_center').value, p('pan_range').value, p('pan_steps').value
        pans = [c - r + 2 * r * i / (n - 1) for i in range(n)]
        self.waypoints = []
        for k, tilt in enumerate(p('tilts').value):
            row = pans if k % 2 == 0 else pans[::-1]  # zig-zag
            self.waypoints += [(pan, tilt) for pan in row]

        self.found = False
        self.i = 0
        self.client = ActionClient(self, FollowJointTrajectory, p('action_name').value)
        self.create_subscription(PointStamped, 'tea_tag/position', self.on_found, 10)
        self.timer = None
        self.get_logger().info('Waiting for trajectory action server...')
        self.client.wait_for_server()
        self.next_waypoint()

    def on_found(self, _msg):
        if not self.found:
            self.found = True
            self.get_logger().info('Tag localized - stopping scan.')

    def next_waypoint(self):
        if self.found:
            return
        if self.i >= len(self.waypoints):
            self.get_logger().warn('Scan finished, tag not found.')
            return
        pan, tilt = self.waypoints[self.i]
        self.i += 1
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
        pt = JointTrajectoryPoint(positions=[pan, tilt])
        t = self.get_parameter('move_time').value
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        goal.trajectory.points = [pt]
        self.get_logger().info(f'Head -> pan {pan:.2f}, tilt {tilt:.2f}')
        self.client.send_goal_async(goal).add_done_callback(self.on_accepted)

    def on_accepted(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected')
            return
        handle.get_result_async().add_done_callback(self.on_done)

    def on_done(self, _fut):
        # dwell so the detector can process a few frames, then move on
        self.timer = self.create_timer(self.get_parameter('dwell').value, self.on_dwell)

    def on_dwell(self):
        self.timer.cancel()
        self.destroy_timer(self.timer)
        self.next_waypoint()


def main(args=None):
    rclpy.init(args=args)
    node = HeadScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
