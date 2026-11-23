"""ROS 2 node: move Stretch's gripper to a hover point above the target object.

Reads the target position from `orange_block/position` (published by
yolo_world_localizer, base_link frame; override with `-p tag_topic:=...`), then:
  1. sets the wrist: level (pitch 0, `approach:=side`, default) or pointing
     down (pitch -1.57, `approach:=top`),
  2. drives the base forward/back so the gripper is level with the tag in x,
  3. moves the lift to the pregrasp height (arm still retracted),
  4. extends the arm until the gripper is `standoff` short of the object (side)
     or `hover` above it (top).

When done it publishes the target (fixed frame, latched) on `pregrasp/target`;
the `grasp` node waits for that message before picking the object up.

DRY RUN by default: prints the plan and does not move. Pass `-p execute:=true`.
Needs the base in navigation mode (the node calls /switch_to_navigation_mode).
"""
import sys
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


def plan_pregrasp(tag_b, grasp_b, lift, ext, offset, arm_axis):
    """Compute motion from positions expressed in base_link.

    base_link: x forward, y left, z up. The arm extends along `arm_axis`
    (Stretch: -y). The grasp centre goes to tag + offset (top approach: straight
    above; side approach: in front of the object, short along the arm axis).
    Returns dict with base_dx, lift_target, ext_target, warnings.
    """
    tag_b, grasp_b = np.asarray(tag_b, float), np.asarray(grasp_b, float)
    arm_axis = np.asarray(arm_axis, float)
    goal = tag_b + np.asarray(offset, float)
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


# joint_states tolerance for "arrived" (m for prismatic, rad for revolute)
JOINT_TOL = {'joint_lift': 0.005, 'wrist_extension': 0.005,
             'joint_wrist_pitch': 0.03, 'joint_wrist_yaw': 0.03, 'joint_wrist_roll': 0.03,
             'joint_head_pan': 0.03, 'joint_head_tilt': 0.03}

# latched, so a grasp node started after pregrasp finished still gets the target
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def sim_time_overrides():
    """Use the sim's /clock unless use_sim_time is given on the command line.

    The MuJoCo sim often runs far below real time, so wall-clock timeouts fire long
    before the robot has moved. On the real robot (no /clock) pass
    `-p use_sim_time:=false`, otherwise ROS time stays at 0 and timeouts never fire.
    """
    cli_sets_sim_time = any('use_sim_time' in a for a in sys.argv)
    return [] if cli_sets_sim_time else [Parameter('use_sim_time', value=True)]


class ArmMotion:
    """TF + trajectory helpers shared by Pregrasp and Grasp.

    Expects: self.tf_buffer, self.base_frame, self.fixed, self.grasp_frame,
    self.client (FollowJointTrajectory ActionClient), self.joints = {} and a
    /stretch/joint_states subscription calling self.on_joints.

    Freshness: in the sim, joint_states / TF are stamped with wall time and go
    through joint_state_publisher + robot_state_publisher, so right after a move
    the latest TF can still show the old pose. Every settle (wait_settled /
    wait_base_stopped) records the newest joint_states stamp, and in_base /
    grasp_in_base wait for TF at least that new before using it.
    """
    js_ns = None       # stamp (ns) of the newest /stretch/joint_states message
    settle_ns = None   # js_ns when the robot last settled

    def on_joints(self, msg):
        self.joints.update(dict(zip(msg.name, msg.position)))
        self.js_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

    def mark_settled(self):
        self.settle_ns = self.js_ns

    def lookup(self, target, source, fresh=True, timeout=5.0):
        """Latest TF target<-source; if fresh, newer than the last settle."""
        deadline = time.monotonic() + timeout
        while True:
            tf = self.tf_buffer.lookup_transform(
                target, source, Time(), RclDuration(seconds=1.0))
            st = tf.header.stamp.sec * 10**9 + tf.header.stamp.nanosec
            if not fresh or self.settle_ns is None or st >= self.settle_ns:
                return tf
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    f'TF {target}<-{source} is {(self.settle_ns - st) * 1e-9:.2f} s older '
                    'than the last settle; using it anyway')
                return tf
            time.sleep(0.05)

    def in_base(self, p_fixed):
        tf = self.lookup(self.base_frame, self.fixed)
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_matrix(q.x, q.y, q.z, q.w) @ p_fixed + [t.x, t.y, t.z]

    def grasp_in_base(self):
        t = self.lookup(self.base_frame, self.grasp_frame).transform.translation
        return np.array([t.x, t.y, t.z])

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
        self.wait_settled(names, positions)
        time.sleep(0.4)  # server rejects goals sent <0.2 s apart

    def wait_settled(self, names, positions, timeout=30.0, stall_secs=1.5):
        """Block until every joint is at its target or has stopped moving.

        The MuJoCo driver reports a trajectory as done once a joint is within 5 cm
        of its target (or after 5 s wall time, while the sim runs slower than real
        time), so the arm can still be moving when the action returns. Waits on
        /stretch/joint_states (self.joints) using ROS time. A joint that stops short
        (e.g. the gripper squeezing an object) counts as settled after stall_secs.
        """
        try:
            self._wait_settled(names, positions, timeout, stall_secs)
        finally:
            self.mark_settled()

    def _wait_settled(self, names, positions, timeout, stall_secs):
        joints = getattr(self, 'joints', None)
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
        self.get_logger().warn(
            f'{names} still moving after {timeout:.0f} s: '
            f'{ {n: round(joints.get(n, float("nan")), 3) for n in targets} }')

    def now_s(self):
        """ROS time in s (sim time when run with -p use_sim_time:=true)."""
        return self.get_clock().now().nanoseconds * 1e-9


class Pregrasp(ArmMotion, Node):
    def __init__(self):
        super().__init__('pregrasp', parameter_overrides=sim_time_overrides())
        # YOLO-World detection (yolo_world_localizer), base_link frame
        self.declare_parameter('tag_topic', 'orange_block/position')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('grasp_frame', 'link_grasp_center')
        self.declare_parameter('action_name', '/stretch_controller/follow_joint_trajectory')
        # 'side': gripper level (pitch 0), stops `standoff` short of the object at
        #         about grasp height; `grasp` then extends straight in.
        # 'top' : gripper pointing down, `hover` above the object.
        self.declare_parameter('approach', 'side')
        self.declare_parameter('hover', 0.10)       # m above tag (top)
        self.declare_parameter('standoff', 0.08)    # m short of the object along the arm (side)
        self.declare_parameter('side_z', 0.02)      # m above the detected centre (side)
        self.declare_parameter('samples', 10)
        self.declare_parameter('arm_axis', [0.0, -1.0, 0.0])
        self.declare_parameter('execute', True)
        self.declare_parameter('base_speed', 0.3)   # m/s top speed (tapers near the goal)
        self.declare_parameter('x_tol', 0.01)       # m, base x alignment tolerance
        self.declare_parameter('min_speed', 0.01)   # m/s near the goal (odom lags -> overshoot)
        self.declare_parameter('drive_tries', 3)    # drive / re-measure / correct rounds

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.base_frame, self.fixed = g('base_frame'), g('fixed_frame')
        self.grasp_frame = g('grasp_frame')
        self.n_samples, self.arm_axis = g('samples'), g('arm_axis')
        self.approach = g('approach')
        if self.approach == 'side':
            self.offset = (-g('standoff') * np.asarray(self.arm_axis, float)
                           + [0.0, 0.0, g('side_z')])
            self.wrist_pitch = 0.0
        elif self.approach == 'top':
            self.offset = np.array([0.0, 0.0, g('hover')])
            self.wrist_pitch = -1.57
        else:
            raise ValueError(f"approach must be 'side' or 'top', got {self.approach!r}")
        self.execute, self.speed = g('execute'), g('base_speed')
        self.x_tol, self.drive_tries = g('x_tol'), g('drive_tries')
        self.min_speed = g('min_speed')

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
        self.target_pub = self.create_publisher(PointStamped, 'pregrasp/target', LATCHED)
        self.started = False
        self.get_logger().info(
            f'Collecting {self.n_samples} tag samples '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    # ---------- callbacks ----------
    def on_tag(self, msg):
        if self.started:
            return
        try:  # store in the fixed frame so it survives base motion
            tf = self.tf_buffer.lookup_transform(
                self.fixed, msg.header.frame_id, Time(), RclDuration(seconds=0.5))
        except Exception as e:
            hint = ''
            if self.fixed == 'odom' and 'does not exist' in str(e):
                hint = (' (real robot: launch stretch_driver with broadcast_odom_tf:=True)')
            self.get_logger().warn(f'TF failed: {e}{hint}', throttle_duration_sec=2.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        p = quat_to_matrix(q.x, q.y, q.z, q.w) @ [msg.point.x, msg.point.y, msg.point.z]
        self.samples.append(p + [t.x, t.y, t.z])
        if len(self.samples) >= self.n_samples:
            self.started = True
            threading.Thread(target=self.run, daemon=True).start()

    # ---------- helpers ----------
    def pose_in_fixed(self, fresh=False):
        tf = self.lookup(self.fixed, self.base_frame, fresh=fresh)
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y]), quat_to_matrix(q.x, q.y, q.z, q.w)

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
        start, R = self.pose_in_fixed(fresh=True)
        fwd = R[:2, 0]
        sign = 1.0 if dx > 0 else -1.0
        self.get_logger().info(f'Driving {dx:+.3f} m along base x')
        t0 = self.now_s()
        while self.now_s() - t0 < abs(dx) / self.speed + 10.0:
            pos, _ = self.pose_in_fixed()
            travelled = float((pos - start) @ fwd)
            if (dx - travelled) * sign <= 0.003:
                break
            if self.now_s() - t0 > 3.0 and abs(travelled) < min(0.01, abs(dx) / 3):
                self.cmd_pub.publish(Twist())
                raise RuntimeError(
                    f'base did not move (odom travelled {travelled:.3f} m after 3 s, '
                    f"driver mode '{self.mode}'). Check the sim terminal for "
                    'cmd_vel / mode errors and for duplicate driver processes.')
            # full speed, tapering (proportional to remaining distance) near the goal
            tw = Twist()
            tw.linear.x = sign * min(self.speed, max(self.min_speed, 1.5 * abs(dx - travelled)))
            self.cmd_pub.publish(tw)
            time.sleep(0.05)
        self.cmd_pub.publish(Twist())
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
                    self.offset, self.arm_axis)

            p = plan()
            self.get_logger().info(f'Plan (current wrist pose): {p}')
            if not self.execute:
                self.get_logger().info('DRY RUN - rerun with -p execute:=true to move.')
                return

            if not self.client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('trajectory action server not available')
            # 0. side approach lowers the lift, so pull the arm in first (a previous
            #    run may have left it extended over the table)
            if self.approach == 'side' and self.joints['wrist_extension'] > 0.05:
                self.move_joints(['wrist_extension'], [0.0])
            # 1. wrist level (side) / straight down (top), so grasp-center offsets are final
            self.move_joints(['joint_wrist_pitch', 'joint_wrist_yaw'], [self.wrist_pitch, 0.0])
            time.sleep(0.5)
            # 2. base x alignment
            self.ensure_navigation_mode()
            self.mark_settled()
            for _ in range(self.drive_tries):
                dx = plan()['base_dx']
                if abs(dx) <= self.x_tol:
                    break
                self.drive(dx)
            # 3. recompute and check reach with the new base pose
            p = plan()
            self.get_logger().info(f'Plan after driving: {p}')
            if abs(p['base_dx']) > max(self.x_tol, 0.015):
                raise RuntimeError(
                    f"base still {p['base_dx']:.3f} m from the tag in x after driving")
            if p['warnings']:
                raise RuntimeError('; '.join(p['warnings']))
            # 4. lift first (arm still retracted, so clear of the table), then extend
            self.move_joints(['joint_lift'], [p['lift_target']])
            self.move_joints(['wrist_extension'], [p['ext_target']])
            final = plan()
            err = np.array(final['goal_base']) - self.grasp_in_base()
            self.get_logger().info(
                f'DONE ({self.approach}). Gripper-to-pregrasp error in base_link (m): '
                f'{np.round(err, 3)}')
            out = PointStamped()
            out.header.frame_id = self.fixed
            out.header.stamp = self.get_clock().now().to_msg()
            out.point.x, out.point.y, out.point.z = (float(c) for c in tag_fixed)
            self.target_pub.publish(out)
            self.get_logger().info('Published pregrasp/target - grasp node may proceed.')
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
