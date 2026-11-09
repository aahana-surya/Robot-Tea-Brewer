"""ROS 2 node: move Stretch's gripper to a hover point above the tea-bag tag.

Reads the tag position from `tea_tag/position` (stage 1), then:
  1. points the wrist down (pitch -1.57, yaw 0),
  2. drives the base forward/back so the gripper is level with the tag in x,
  3. raises the lift to the hover height (clear of the table),
  4. extends the arm until the gripper is above the tag.

DRY RUN by default: prints the plan and does not move. Pass `-p execute:=true`.
Needs the base in navigation mode (the node calls /switch_to_navigation_mode).
"""
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, Twist
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_tea_brewer.tag_localizer import quat_to_matrix

LIFT_RANGE = (0.0, 1.1)
EXT_RANGE = (0.0, 0.52)


def plan_pregrasp(tag_b, grasp_b, lift, ext, hover, arm_axis):
    """Compute motion from positions expressed in base_link.

    base_link: x forward, y left, z up. The arm extends along `arm_axis`
    (Stretch: -y). Returns dict with base_dx, lift_target, ext_target, warnings.
    """
    tag_b, grasp_b = np.asarray(tag_b, float), np.asarray(grasp_b, float)
    arm_axis = np.asarray(arm_axis, float)
    goal = tag_b + [0.0, 0.0, hover]
    err = goal - grasp_b
    dx = float(err[0])                           # drive along base x
    lift_t = lift + float(err[2])
    ext_t = ext + float(arm_axis @ err)
    warnings = []
    if not LIFT_RANGE[0] <= lift_t <= LIFT_RANGE[1]:
        warnings.append(f'lift target {lift_t:.3f} outside {LIFT_RANGE}')
    if not EXT_RANGE[0] <= ext_t <= EXT_RANGE[1]:
        warnings.append(f'arm extension target {ext_t:.3f} outside {EXT_RANGE}'
                        ' - base is too far from / too close to the tag')
    return {'base_dx': dx, 'lift_target': lift_t, 'ext_target': ext_t,
            'goal_base': goal.tolist(), 'warnings': warnings}


class Pregrasp(Node):
    def __init__(self):
        super().__init__('pregrasp')
        self.declare_parameter('tag_topic', 'tea_tag/position')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('grasp_frame', 'link_grasp_center')
        self.declare_parameter('action_name', '/stretch_controller/follow_joint_trajectory')
        self.declare_parameter('hover', 0.10)       # m above tag
        self.declare_parameter('samples', 10)
        self.declare_parameter('arm_axis', [0.0, -1.0, 0.0])
        self.declare_parameter('execute', True)
        self.declare_parameter('base_speed', 0.08)  # m/s

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.base_frame, self.fixed = g('base_frame'), g('fixed_frame')
        self.grasp_frame, self.hover = g('grasp_frame'), g('hover')
        self.n_samples, self.arm_axis = g('samples'), g('arm_axis')
        self.execute, self.speed = g('execute'), g('base_speed')

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.samples = []
        self.joints = {}
        self.mode = None
        self.create_subscription(PointStamped, g('tag_topic'), self.on_tag, 10)
        self.create_subscription(JointState, '/stretch/joint_states', self.on_joints, 10)
        self.create_subscription(String, '/mode', lambda m: setattr(self, 'mode', m.data), 10)
        self.cmd_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.client = ActionClient(self, FollowJointTrajectory, g('action_name'))
        self.mode_cli = self.create_client(Trigger, '/switch_to_navigation_mode')
        self.started = False
        self.get_logger().info(
            f'Collecting {self.n_samples} tag samples '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    # ---------- callbacks ----------
    def on_joints(self, msg):
        self.joints.update(dict(zip(msg.name, msg.position)))

    def on_tag(self, msg):
        if self.started:
            return
        try:  # store in the fixed frame so it survives base motion
            tf = self.tf_buffer.lookup_transform(
                self.fixed, msg.header.frame_id, Time(), RclDuration(seconds=0.5))
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}', throttle_duration_sec=2.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        p = quat_to_matrix(q.x, q.y, q.z, q.w) @ [msg.point.x, msg.point.y, msg.point.z]
        self.samples.append(p + [t.x, t.y, t.z])
        if len(self.samples) >= self.n_samples:
            self.started = True
            threading.Thread(target=self.run, daemon=True).start()

    # ---------- helpers ----------
    def in_base(self, p_fixed):
        tf = self.tf_buffer.lookup_transform(
            self.base_frame, self.fixed, Time(), RclDuration(seconds=1.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_matrix(q.x, q.y, q.z, q.w) @ p_fixed + [t.x, t.y, t.z]

    def grasp_in_base(self):
        tf = self.tf_buffer.lookup_transform(
            self.base_frame, self.grasp_frame, Time(), RclDuration(seconds=1.0))
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def pose_in_fixed(self):
        tf = self.tf_buffer.lookup_transform(
            self.fixed, self.base_frame, Time(), RclDuration(seconds=1.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y]), quat_to_matrix(q.x, q.y, q.z, q.w)

    def move_joints(self, names, positions, secs=3.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        pt = JointTrajectoryPoint(positions=[float(x) for x in positions])
        pt.time_from_start = Duration(sec=int(secs))
        goal.trajectory.points = [pt]
        self.get_logger().info(f'Move {dict(zip(names, np.round(positions, 3)))}')
        gh_fut = self.client.send_goal_async(goal)
        while not gh_fut.done():
            time.sleep(0.05)
        gh = gh_fut.result()
        if not gh.accepted:
            raise RuntimeError('trajectory goal rejected')
        res_fut = gh.get_result_async()
        while not res_fut.done():
            time.sleep(0.05)
        time.sleep(0.4)  # server rejects goals sent <0.2 s apart

    def ensure_navigation_mode(self):
        for _ in range(5):
            if self.mode == 'navigation':
                return
            if self.mode_cli.wait_for_service(timeout_sec=2.0):
                self.mode_cli.call_async(Trigger.Request())
            time.sleep(1.0)
        raise RuntimeError(
            f"driver /mode is '{self.mode}', not 'navigation'. If several "
            'stretch_mujoco_driver processes are running, kill the stale ones.')

    def drive(self, dx):
        if abs(dx) < 0.005:
            return
        start, R = self.pose_in_fixed()
        fwd = R[:2, 0]
        sign = 1.0 if dx > 0 else -1.0
        tw = Twist()
        tw.linear.x = sign * self.speed
        self.get_logger().info(f'Driving {dx:+.3f} m along base x')
        t0 = time.time()
        while time.time() - t0 < abs(dx) / self.speed + 10.0:
            pos, _ = self.pose_in_fixed()
            travelled = float((pos - start) @ fwd)
            if (dx - travelled) * sign <= 0.003:
                break
            if time.time() - t0 > 3.0 and abs(travelled) < 0.01:
                self.cmd_pub.publish(Twist())
                raise RuntimeError(
                    f'base did not move (odom travelled {travelled:.3f} m after 3 s, '
                    f"driver mode '{self.mode}'). Check the sim terminal for "
                    'cmd_vel / mode errors and for duplicate driver processes.')
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.cmd_pub.publish(Twist())
        time.sleep(0.5)

    # ---------- main sequence ----------
    def run(self):
        try:
            tag_fixed = np.median(np.array(self.samples), axis=0)
            self.get_logger().info(f'Tag in {self.fixed}: {np.round(tag_fixed, 3)}')
            if not self.joints:
                raise RuntimeError('no /stretch/joint_states yet')

            def plan():
                return plan_pregrasp(
                    self.in_base(tag_fixed), self.grasp_in_base(),
                    self.joints['joint_lift'], self.joints['wrist_extension'],
                    self.hover, self.arm_axis)

            p = plan()
            self.get_logger().info(f'Plan (current wrist pose): {p}')
            if not self.execute:
                self.get_logger().info('DRY RUN - rerun with -p execute:=true to move.')
                return

            if not self.client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('trajectory action server not available')
            # 1. wrist straight down, so grasp-center offsets are final
            self.move_joints(['joint_wrist_pitch', 'joint_wrist_yaw'], [-1.57, 0.0])
            time.sleep(0.5)
            # 2. base x alignment
            self.ensure_navigation_mode()
            self.drive(plan()['base_dx'])
            # 3. recompute and check reach with the new base pose
            p = plan()
            self.get_logger().info(f'Plan after driving: {p}')
            if abs(p['base_dx']) > 0.03:
                raise RuntimeError(
                    f"base still {p['base_dx']:.3f} m from the tag in x after driving")
            if p['warnings']:
                raise RuntimeError('; '.join(p['warnings']))
            # 4. lift first (clears the table), then extend
            self.move_joints(['joint_lift'], [p['lift_target']])
            self.move_joints(['wrist_extension'], [p['ext_target']])
            time.sleep(0.8)
            final = plan()
            err = np.array(final['goal_base']) - self.grasp_in_base()
            self.get_logger().info(
                f'DONE. Gripper-to-hover error in base_link (m): {np.round(err, 3)}')
        except Exception as e:
            self.get_logger().error(f'Pregrasp aborted: {e}')
        finally:
            self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Pregrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
