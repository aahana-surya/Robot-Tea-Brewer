"""Reusable Stretch motion primitives shared by pregrasp, grasp, steep_tea, etc.

Nothing here is a Node. Both classes wrap an existing node, so a future
orchestrator can create ONE node and hand it to several behaviors:

  StretchMotion  - TF helpers, joint trajectory moves (waiting until the joints
                   have actually settled), base driving, mode switching, and
                   `go_to()` / `go_above()` (bring the gripper to a pregrasp /
                   hover point next to a 3D target).
  TargetSampler  - collects N samples of a PointStamped topic (e.g.
                   orange_block/position, cup/position), stores them in the fixed
                   frame so they survive base motion, and returns the median.

All blocking calls (move_joints, drive, go_to) must run on a worker thread,
not in a subscription callback, since they wait on futures while the node spins.

Freshness: in the sim, joint_states / TF are stamped with wall time and go
through joint_state_publisher + robot_state_publisher, so right after a move
the latest TF can still show the old pose. Every settle (wait_settled /
wait_base_stopped) records the newest joint_states stamp, and in_base /
grasp_in_base wait for TF at least that new before using it.
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
from rclpy.qos import DurabilityPolicy, QoSProfile
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

# joint_states tolerance for "arrived" (m for prismatic, rad for revolute)
JOINT_TOL = {'joint_lift': 0.005, 'wrist_extension': 0.005,
             'joint_wrist_pitch': 0.03, 'joint_wrist_yaw': 0.03, 'joint_wrist_roll': 0.03,
             'joint_head_pan': 0.03, 'joint_head_tilt': 0.03}

# latched, so a node started after the publisher finished still gets the message
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


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
    d('min_speed', 0.01)   # m/s near the goal (odom lags -> overshoot)
    d('x_tol', 0.01)       # m, base x alignment tolerance
    d('drive_tries', 3)    # drive / re-measure / correct rounds


def plan_hover(target_b, grasp_b, lift, ext, offset, arm_axis):
    """Compute the motion that puts the grasp center at target + offset.

    All positions are expressed in base_link (x forward, y left, z up). The arm
    extends along `arm_axis` (Stretch: -y). `offset` is a 3-vector (e.g. side
    approach: short of the object along the arm) or a scalar hover height
    (straight above). Returns dict with base_dx, lift_target, ext_target,
    goal_base, warnings.
    """
    target_b, grasp_b = np.asarray(target_b, float), np.asarray(grasp_b, float)
    arm_axis = np.asarray(arm_axis, float)
    offset = np.asarray(offset, float)
    if offset.ndim == 0:
        offset = np.array([0.0, 0.0, float(offset)])
    goal = target_b + offset
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
        self.speed, self.min_speed = g('base_speed'), g('min_speed')
        self.x_tol, self.drive_tries = g('x_tol'), g('drive_tries')
        self.log = node.get_logger()

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, node)
        self.joints = {}
        self.js_ns = None       # stamp (ns) of the newest /stretch/joint_states message
        self.settle_ns = None   # js_ns when the robot last settled
        self.mode = None
        node.create_subscription(JointState, '/stretch/joint_states', self._on_joints, 10)
        node.create_subscription(String, '/mode', lambda m: setattr(self, 'mode', m.data), 10)
        self.cmd_pub = node.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.client = ActionClient(node, FollowJointTrajectory, g('action_name'))
        self.mode_cli = node.create_client(Trigger, '/switch_to_navigation_mode')

    def _on_joints(self, msg):
        self.joints.update(dict(zip(msg.name, msg.position)))
        self.js_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

    def stop(self):
        self.cmd_pub.publish(Twist())

    def now_s(self):
        """ROS time in s (sim time when run with -p use_sim_time:=true)."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def wait_for_joints(self, timeout=5.0):
        t0 = time.monotonic()
        while not self.joints and time.monotonic() - t0 < timeout:
            time.sleep(0.1)
        if not self.joints:
            raise RuntimeError('no /stretch/joint_states yet')

    def mark_settled(self):
        self.settle_ns = self.js_ns

    # ---------- TF ----------
    def _tf(self, target, source, timeout=1.0, fresh=False, fresh_timeout=5.0):
        """Latest TF target<-source as (t, R); if fresh, newer than the last settle."""
        deadline = time.monotonic() + fresh_timeout
        while True:
            tf = self.tf_buffer.lookup_transform(
                target, source, Time(), RclDuration(seconds=timeout))
            st = tf.header.stamp.sec * 10**9 + tf.header.stamp.nanosec
            if not fresh or self.settle_ns is None or st >= self.settle_ns:
                break
            if time.monotonic() > deadline:
                self.log.warn(
                    f'TF {target}<-{source} is {(self.settle_ns - st) * 1e-9:.2f} s older '
                    'than the last settle; using it anyway')
                break
            time.sleep(0.05)
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), quat_to_matrix(q.x, q.y, q.z, q.w)

    def in_base(self, p_fixed):
        t, R = self._tf(self.base_frame, self.fixed, fresh=True)
        return R @ p_fixed + t

    def grasp_in_base(self):
        return self._tf(self.base_frame, self.grasp_frame, fresh=True)[0]

    def pose_in_fixed(self, fresh=False):
        t, R = self._tf(self.fixed, self.base_frame, fresh=fresh)
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
        self.wait_settled(names, positions)
        time.sleep(0.4)  # server rejects goals sent <0.2 s apart

    def wait_settled(self, names, positions, timeout=30.0, stall_secs=1.5):
        """Block until every joint is at its target or has stopped moving.

        The MuJoCo driver reports a trajectory as done once a joint is within 5 cm
        of its target (or after 5 s wall time, while the sim runs slower than real
        time), so the arm can still be moving when the action returns. Waits on
        /stretch/joint_states using ROS time. A joint that stops short (e.g. the
        gripper squeezing an object) counts as settled after stall_secs.
        """
        try:
            self._wait_settled(names, positions, timeout, stall_secs)
        finally:
            self.mark_settled()

    def _wait_settled(self, names, positions, timeout, stall_secs):
        joints = self.joints
        if not joints:
            return
        targets = dict(zip(names, (float(p) for p in positions)))
        t0 = last_move = self.now_s()
        last = {n: joints.get(n) for n in targets}
        while self.now_s() - t0 < timeout:
            cur = {n: joints.get(n) for n in targets}
            if None in cur.values():
                return  # joint not reported in joint_states; nothing to wait on
            if all(abs(cur[n] - targets[n]) <= JOINT_TOL.get(n, 0.02) for n in targets):
                return
            if any(abs(cur[n] - last[n]) > 0.001 for n in targets):
                last, last_move = cur, self.now_s()
            elif self.now_s() - last_move > stall_secs:
                return  # stopped short of the target (contact / limit)
            time.sleep(0.05)
        self.log.warn(
            f'{names} still moving after {timeout:.0f} s: '
            f'{ {n: round(joints.get(n, float("nan")), 3) for n in targets} }')

    def wait_for_server(self, timeout=5.0):
        if not self.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('trajectory action server not available')

    def set_wrist(self, pitch, yaw=0.0, roll=None):
        """Orient the wrist (roll left alone when None), so grasp-center offsets are final."""
        names, pos = ['joint_wrist_pitch', 'joint_wrist_yaw'], [pitch, yaw]
        if roll is not None:
            names.append('joint_wrist_roll')
            pos.append(roll)
        self.move_joints(names, pos)
        time.sleep(0.5)

    def wrist_down(self):
        """Point the wrist straight down so grasp-center offsets are final."""
        self.set_wrist(WRIST_DOWN, 0.0)

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
        start, R = self.pose_in_fixed(fresh=True)
        fwd = R[:2, 0]
        sign = 1.0 if dx > 0 else -1.0
        self.log.info(f'Driving {dx:+.3f} m along base x')
        t0 = self.now_s()
        while self.now_s() - t0 < abs(dx) / self.speed + 10.0:
            pos, _ = self.pose_in_fixed()
            travelled = float((pos - start) @ fwd)
            if (dx - travelled) * sign <= 0.003:
                break
            if self.now_s() - t0 > 3.0 and abs(travelled) < min(0.01, abs(dx) / 3):
                self.stop()
                raise RuntimeError(
                    f'base did not move (odom travelled {travelled:.3f} m after 3 s, '
                    f"driver mode '{self.mode}'). Check the sim terminal for "
                    'cmd_vel / mode errors and for duplicate driver processes.')
            # full speed, tapering (proportional to remaining distance) near the goal
            tw = Twist()
            tw.linear.x = sign * min(self.speed, max(self.min_speed, 1.5 * abs(dx - travelled)))
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.stop()
        self.wait_base_stopped()

    def wait_base_stopped(self, still_secs=1.0, timeout=20.0):
        """Wait (ROS time) until odom stops changing, then mark the robot settled."""
        t0 = last_move = self.now_s()
        last, _ = self.pose_in_fixed()
        while self.now_s() - t0 < timeout:
            time.sleep(0.1)
            pos, _ = self.pose_in_fixed()
            if np.linalg.norm(pos - last) > 0.001:
                last, last_move = pos, self.now_s()
            elif self.now_s() - last_move > still_secs:
                break
        self.mark_settled()

    # ---------- composite ----------
    def plan_to(self, target_fixed, offset):
        return plan_hover(self.in_base(np.asarray(target_fixed, float)), self.grasp_in_base(),
                          self.joints['joint_lift'], self.joints['wrist_extension'],
                          offset, self.arm_axis)

    def go_to(self, target_fixed, offset, wrist=(WRIST_DOWN, 0.0, None),
              raise_before_drive=False, retract_first=False):
        """Bring the grasp center to `target_fixed` (fixed frame) + `offset` (base_link).

        (optional retract) -> wrist -> (optional raise) -> drive base x (with
        re-measure / correct rounds) -> lift -> extend.
        `wrist` is (pitch, yaw, roll); roll None leaves it alone.
        `raise_before_drive` lifts first when the gripper holds something that
        should clear the table before the base moves. `retract_first` pulls the
        arm in before anything else (e.g. before lowering the lift next to a
        table). Returns the residual (goal - grasp) error in base_link.
        """
        target_fixed = np.asarray(target_fixed, float)
        if not self.joints:
            raise RuntimeError('no /stretch/joint_states yet')

        def plan():
            return self.plan_to(target_fixed, offset)

        self.wait_for_server()
        if retract_first and self.joints['wrist_extension'] > 0.05:
            self.move_joints(['wrist_extension'], [0.0])
        self.set_wrist(*wrist)
        self.ensure_navigation_mode()
        self.mark_settled()
        p = plan()
        self.log.info(f'Plan (current wrist pose): {p}')
        if raise_before_drive and p['lift_target'] > self.joints['joint_lift']:
            self.move_joints(['joint_lift'], [min(p['lift_target'], LIFT_RANGE[1])])
        for _ in range(self.drive_tries):
            dx = plan()['base_dx']
            if abs(dx) <= self.x_tol:
                break
            self.drive(dx)
        p = plan()
        self.log.info(f'Plan after driving: {p}')
        if abs(p['base_dx']) > max(self.x_tol, 0.015):
            raise RuntimeError(f"base still {p['base_dx']:.3f} m from the target in x after driving")
        if p['warnings']:
            raise RuntimeError('; '.join(p['warnings']))
        # lift first (arm still retracted, so clear of the table), then extend
        self.move_joints(['joint_lift'], [p['lift_target']])
        self.move_joints(['wrist_extension'], [p['ext_target']])
        err = np.array(plan()['goal_base']) - self.grasp_in_base()
        self.log.info(f'Gripper-to-goal error in base_link (m): {np.round(err, 3)}')
        return err

    def go_above(self, target_fixed, hover, raise_before_drive=False,
                 wrist=(WRIST_DOWN, 0.0, None)):
        """Grasp center `hover` m straight above `target_fixed` (wrist down by default)."""
        return self.go_to(target_fixed, hover, wrist=wrist,
                          raise_before_drive=raise_before_drive)


class TargetSampler:
    """Collect `n` samples of a PointStamped topic and take their median.

    Samples are transformed into `motion.fixed` so they stay valid while the
    base moves.

    Callback mode (`on_ready` given): starts collecting immediately and calls
    `on_ready(median)` on a new thread (it may block). Call `reset()` to collect
    a fresh set, e.g. after the scene changed.

    Blocking mode (`on_ready=None`): idle until `collect()` is called from a
    worker thread, which returns the median of the next `n` samples (or None on
    timeout).
    """

    def __init__(self, node, motion, topic, n, on_ready=None):
        self.node, self.motion, self.n, self.on_ready = node, motion, n, on_ready
        self.topic = topic
        self.samples = []
        self.started = on_ready is None    # blocking mode: ignore messages until collect()
        self.skip = 0
        self.done = threading.Event()
        node.create_subscription(PointStamped, topic, self._on_point, 10)

    def reset(self):
        self.samples, self.started = [], False

    def collect(self, timeout=30.0, skip=1):
        """Blocking: median of the next `n` samples, or None after `timeout` s (wall).

        `skip` drops the first messages received: a detection already being
        computed when collect() is called may come from an image taken while the
        robot was still moving.
        """
        self.done.clear()
        self.skip = skip
        self.reset()
        if not self.done.wait(timeout):
            self.started = True
            got = len(self.samples)
            self.node.get_logger().warn(
                f'{self.topic}: only {got}/{self.n} samples after {timeout:.0f} s')
            return None
        return self.median

    def _on_point(self, msg):
        if self.started:
            return
        if self.skip > 0:
            self.skip -= 1
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
            self.median = np.median(np.array(self.samples), axis=0)
            self.done.set()
            if self.on_ready is not None:
                threading.Thread(target=self.on_ready, args=(self.median,), daemon=True).start()
