"""Reusable Stretch motion primitives shared by pregrasp, steep_tea, etc.

Nothing here is a Node. Both classes wrap an existing node, so a future
orchestrator can create ONE node and hand it to several behaviors:

  StretchMotion  - TF helpers, joint trajectory moves, base driving, mode
                   switching, and `go_above()` (bring the gripper to a hover
                   point above a 3D target).
  TargetSampler  - collects N samples of a PointStamped topic (e.g.
                   tea_bag/position, cup/position), stores them in the fixed
                   frame so they survive base motion, and returns the median.

All blocking calls (move_joints, drive, go_above) must run on a worker thread,
not in a subscription callback, since they wait on futures while the node spins.
"""
import sys
import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, Twist
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_tea_brewer.tag_localizer import quat_to_matrix

LIFT_RANGE = (0.0, 1.1)
EXT_RANGE = (0.0, 0.52)
WRIST_DOWN = -1.57


def sim_time_overrides():
    """Parameter overrides for Node(..., parameter_overrides=...).

    Use the sim's /clock by default: the MuJoCo sim often runs far below real
    time, so wall-clock timeouts fire long before the robot has moved. On the
    real robot (no /clock) pass `-p use_sim_time:=false`, otherwise ROS time
    stays at 0 and the drive timeouts / stall check never fire.
    """
    if any('use_sim_time' in a for a in sys.argv):
        return []
    return [Parameter('use_sim_time', value=True)]


def declare_motion_parameters(node):
    """Declare the parameters StretchMotion reads (call before constructing it)."""
    d = node.declare_parameter
    d('base_frame', 'base_link')
    d('fixed_frame', 'odom')
    d('grasp_frame', 'link_grasp_center')
    d('action_name', '/stretch_controller/follow_joint_trajectory')
    d('arm_axis', [0.0, -1.0, 0.0])
    d('base_speed', 0.3)   # m/s top speed (tapers near the goal)


def plan_hover(target_b, grasp_b, lift, ext, hover, arm_axis):
    """Compute the motion that puts the grasp center `hover` m above a target.

    All positions are expressed in base_link (x forward, y left, z up). The arm
    extends along `arm_axis` (Stretch: -y). Returns dict with base_dx,
    lift_target, ext_target, goal_base, warnings.
    """
    target_b, grasp_b = np.asarray(target_b, float), np.asarray(grasp_b, float)
    arm_axis = np.asarray(arm_axis, float)
    goal = target_b + [0.0, 0.0, hover]
    err = goal - grasp_b
    dx = float(err[0])                           # drive along base x
    lift_t = lift + float(err[2])
    ext_t = ext + float(arm_axis @ err)
    warnings = []
    if not LIFT_RANGE[0] <= lift_t <= LIFT_RANGE[1]:
        warnings.append(f'lift target {lift_t:.3f} outside {LIFT_RANGE}')
    if not EXT_RANGE[0] <= ext_t <= EXT_RANGE[1]:
        warnings.append(f'arm extension target {ext_t:.3f} outside {EXT_RANGE}'
                        ' - base is too far from / too close to the target')
    return {'base_dx': dx, 'lift_target': lift_t, 'ext_target': ext_t,
            'goal_base': goal.tolist(), 'warnings': warnings}


class StretchMotion:
    def __init__(self, node):
        """`node` must have had declare_motion_parameters(node) called."""
        self.node = node
        g = lambda n: node.get_parameter(n).value  # noqa: E731
        self.base_frame, self.fixed = g('base_frame'), g('fixed_frame')
        self.grasp_frame, self.arm_axis = g('grasp_frame'), g('arm_axis')
        self.speed = g('base_speed')
        self.log = node.get_logger()

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, node)
        self.joints = {}
        self.mode = None
        node.create_subscription(JointState, '/stretch/joint_states', self._on_joints, 10)
        node.create_subscription(String, '/mode', lambda m: setattr(self, 'mode', m.data), 10)
        self.cmd_pub = node.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.client = ActionClient(node, FollowJointTrajectory, g('action_name'))
        self.mode_cli = node.create_client(Trigger, '/switch_to_navigation_mode')

    def _on_joints(self, msg):
        self.joints.update(dict(zip(msg.name, msg.position)))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def now_s(self):
        """ROS time in s (sim time when run with -p use_sim_time:=true)."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    # ---------- TF ----------
    def _tf(self, target, source, timeout=1.0):
        tf = self.tf_buffer.lookup_transform(target, source, Time(), RclDuration(seconds=timeout))
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), quat_to_matrix(q.x, q.y, q.z, q.w)

    def in_base(self, p_fixed):
        t, R = self._tf(self.base_frame, self.fixed)
        return R @ p_fixed + t

    def grasp_in_base(self):
        return self._tf(self.base_frame, self.grasp_frame)[0]

    def pose_in_fixed(self):
        t, R = self._tf(self.fixed, self.base_frame)
        return t[:2], R

    # ---------- joints ----------
    def move_joints(self, names, positions, secs=3.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        pt = JointTrajectoryPoint(positions=[float(x) for x in positions])
        pt.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))
        goal.trajectory.points = [pt]
        self.log.info(f'Move {dict(zip(names, np.round(positions, 3)))}')
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

    def wait_for_server(self, timeout=5.0):
        if not self.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('trajectory action server not available')

    def wrist_down(self):
        """Point the wrist straight down so grasp-center offsets are final."""
        self.move_joints(['joint_wrist_pitch', 'joint_wrist_yaw'], [WRIST_DOWN, 0.0])
        time.sleep(0.5)

    # ---------- base ----------
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
        """Drive `dx` m along base x (odom-feedback, tapering near the goal)."""
        if abs(dx) < 0.005:
            return
        start, R = self.pose_in_fixed()
        fwd = R[:2, 0]
        sign = 1.0 if dx > 0 else -1.0
        self.log.info(f'Driving {dx:+.3f} m along base x')
        t0 = self.now_s()
        while self.now_s() - t0 < abs(dx) / self.speed + 10.0:
            pos, _ = self.pose_in_fixed()
            travelled = float((pos - start) @ fwd)
            if (dx - travelled) * sign <= 0.003:
                break
            if self.now_s() - t0 > 3.0 and abs(travelled) < 0.01:
                self.stop()
                raise RuntimeError(
                    f'base did not move (odom travelled {travelled:.3f} m after 3 s, '
                    f"driver mode '{self.mode}'). Check the sim terminal for "
                    'cmd_vel / mode errors and for duplicate driver processes.')
            tw = Twist()
            tw.linear.x = sign * min(self.speed, max(0.03, 1.5 * abs(dx - travelled)))
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.stop()
        time.sleep(0.5)

    # ---------- composite ----------
    def go_above(self, target_fixed, hover, raise_before_drive=False):
        """Bring the grasp center to `hover` m above `target_fixed` (fixed frame).

        Wrist down -> (optional raise) -> drive base x -> lift -> extend.
        `raise_before_drive` lifts first when the gripper holds something that
        should clear the table before the base moves. Returns the residual
        (goal - grasp) error in base_link.
        """
        target_fixed = np.asarray(target_fixed, float)
        if not self.joints:
            raise RuntimeError('no /stretch/joint_states yet')

        def plan():
            return plan_hover(
                self.in_base(target_fixed), self.grasp_in_base(),
                self.joints['joint_lift'], self.joints['wrist_extension'],
                hover, self.arm_axis)

        self.wait_for_server()
        self.wrist_down()
        self.ensure_navigation_mode()
        p = plan()
        self.log.info(f'Plan (current wrist pose): {p}')
        if raise_before_drive and p['lift_target'] > self.joints['joint_lift']:
            self.move_joints(['joint_lift'], [min(p['lift_target'], LIFT_RANGE[1])])
        self.drive(plan()['base_dx'])
        p = plan()
        self.log.info(f'Plan after driving: {p}')
        if abs(p['base_dx']) > 0.05:
            raise RuntimeError(f"base still {p['base_dx']:.3f} m from the target in x after driving")
        if p['warnings']:
            raise RuntimeError('; '.join(p['warnings']))
        # lift first (clears the table), then extend
        self.move_joints(['joint_lift'], [p['lift_target']])
        self.move_joints(['wrist_extension'], [p['ext_target']])
        time.sleep(0.8)
        err = np.array(plan()['goal_base']) - self.grasp_in_base()
        self.log.info(f'Gripper-to-hover error in base_link (m): {np.round(err, 3)}')
        return err


class TargetSampler:
    """Collect `n` samples of a PointStamped topic, then call `on_ready(median)`.

    Samples are transformed into `motion.fixed` so they stay valid while the
    base moves. `on_ready` runs on a new thread (it may block). Call `reset()`
    to collect a fresh set, e.g. after the scene changed.
    """

    def __init__(self, node, motion, topic, n, on_ready):
        self.node, self.motion, self.n, self.on_ready = node, motion, n, on_ready
        self.samples = []
        self.started = False
        node.create_subscription(PointStamped, topic, self._on_point, 10)

    def reset(self):
        self.samples, self.started = [], False

    def _on_point(self, msg):
        if self.started:
            return
        try:
            t, R = self.motion._tf(self.motion.fixed, msg.header.frame_id, timeout=0.5)
        except Exception as e:
            hint = ''
            if self.motion.fixed == 'odom' and 'does not exist' in str(e):
                hint = ' (real robot: launch stretch_driver with broadcast_odom_tf:=True)'
            self.node.get_logger().warn(f'TF failed: {e}{hint}', throttle_duration_sec=2.0)
            return
        self.samples.append(R @ [msg.point.x, msg.point.y, msg.point.z] + t)
        if len(self.samples) >= self.n:
            self.started = True
            threading.Thread(
                target=self.on_ready, args=(np.median(np.array(self.samples), axis=0),),
                daemon=True).start()
