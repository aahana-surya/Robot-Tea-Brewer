"""ROS 2 node: pick up the orange block once the gripper is at the pregrasp pose.

The block stands upright. Sim block: 1 x 1 in square base, 3 in tall
(0.0254 x 0.0254 x 0.0762 m) - 2x the real 0.5 x 0.5 x 1.5 in block; for the real
one pass -p obj_width:=0.0127 -p obj_height:=0.0381. The fingers close across
the square (along base x), `grasp_depth` below the block top.

Waits for `pregrasp/target` (latched PointStamped in the fixed frame, published
by the `pregrasp` node when it is done). Use the same `approach` as pregrasp.

approach:=side (default) - gripper level, block taken from the side:
  1. opens the gripper, wrist level (pitch 0),
  2. sets the lift to grasp height while still `standoff` short of the block,
  3. extends straight in so the block slides between the fingers,
  4. closes the gripper (it stalls on the block), 5. lifts by `lift_height`.

approach:=top - gripper pointing down: aligns the extension over the block,
then lowers the lift onto it, closes and lifts (pushes on tall blocks).

DRY RUN with `-p execute:=false`: prints the plan and does not move.
Run it alongside / after `pregrasp`; it starts as soon as the target arrives.
"""
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from robot_tea_brewer.pregrasp import (EXT_RANGE, LATCHED, LIFT_RANGE, ArmMotion,
                                       sim_time_overrides)

IN = 0.0254
GRIPPER = 'joint_gripper_finger_left'   # works in sim and on the real robot


class Grasp(ArmMotion, Node):
    def __init__(self):
        super().__init__('grasp', parameter_overrides=sim_time_overrides())
        d = self.declare_parameter
        d('target_topic', 'pregrasp/target')
        d('base_frame', 'base_link')
        d('grasp_frame', 'link_grasp_center')
        d('action_name', '/stretch_controller/follow_joint_trajectory')
        d('arm_axis', [0.0, -1.0, 0.0])
        d('execute', True)
        # block size, metres: square base side, height (standing upright)
        d('obj_width', 1.0 * IN)     # sim block (real: 0.5 in)
        d('obj_height', 3.0 * IN)    # sim block (real: 1.5 in)
        d('grasp_depth', 0.02)       # m, grasp centre below the block top
        d('max_xy_err', 0.015)       # m, abort if the gripper is further off the block
        d('table_clearance', 0.004)  # m, keep grasp centre this far above the block bottom
        approach = d('approach', 'side').value      # 'side' or 'top', as in pregrasp
        if approach not in ('side', 'top'):
            raise ValueError(f"approach must be 'side' or 'top', got {approach!r}")
        self.approach = approach
        d('wrist_pitch', 0.0 if approach == 'side' else -1.57)  # level / pointing down
        d('y_tol', 0.01)             # m, max extension error before closing
        # side: push the grasp centre this far PAST the block centre along the arm.
        # link_grasp_center sits near the fingertips, so without this only the tips
        # touch the block.
        d('insert_depth', 0.02)
        d('wrist_yaw', 0.0)          # square base, so yaw only matters if the block is rotated
        d('gripper_open', 0.4)       # joint_gripper_finger_left (rad-ish units)
        d('gripper_close', -0.2)     # past contact -> squeezes the block
        d('lift_height', 0.10)       # m to raise the block after grasping
        d('approach_secs', 4.0)
        d('z_tol', 0.01)             # m, max grasp-centre height error before closing

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.base_frame, self.grasp_frame = g('base_frame'), g('grasp_frame')
        self.fixed = None             # taken from the target message header
        self.arm_axis = np.asarray(g('arm_axis'), float)
        self.execute = g('execute')

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.joints = {}
        self.started = False
        self.client = ActionClient(self, FollowJointTrajectory, g('action_name'))
        self.create_subscription(JointState, '/stretch/joint_states', self.on_joints, 10)
        self.create_subscription(PointStamped, g('target_topic'), self.on_target, LATCHED)
        self.get_logger().info(
            f'Waiting for {g("target_topic")} from pregrasp '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    def on_target(self, msg):
        if self.started:
            return
        self.started = True
        self.fixed = msg.header.frame_id
        target = np.array([msg.point.x, msg.point.y, msg.point.z])
        threading.Thread(target=self.run, args=(target,), daemon=True).start()

    def plan(self, target_fixed):
        """Lift / extension targets that put the grasp centre inside the block."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        obj = self.in_base(target_fixed)          # approx. block centre
        grip = self.grasp_in_base()
        half_h = g('obj_height') / 2
        top = obj[2] + half_h
        # grasp centre: grasp_depth below the top, but not into the table
        z_goal = max(top - g('grasp_depth'), obj[2] - half_h + g('table_clearance'))
        goal = np.array([obj[0], obj[1], z_goal])
        if self.approach == 'side':
            goal = goal + g('insert_depth') * self.arm_axis
        err = goal - grip
        lift_t = self.joints['joint_lift'] + float(err[2])
        ext_t = self.joints['wrist_extension'] + float(self.arm_axis @ err)
        warnings = []
        if abs(err[0]) > g('max_xy_err'):
            warnings.append(f'gripper {err[0]:+.3f} m off the block in base x '
                            '(re-run pregrasp)')
        if not LIFT_RANGE[0] <= lift_t <= LIFT_RANGE[1]:
            warnings.append(f'lift target {lift_t:.3f} outside {LIFT_RANGE}')
        if not EXT_RANGE[0] <= ext_t <= EXT_RANGE[1]:
            warnings.append(f'extension target {ext_t:.3f} outside {EXT_RANGE}')
        return {'obj_base': np.round(obj, 3).tolist(), 'goal_base': goal.tolist(),
                'x_err': float(err[0]), 'lift_target': lift_t, 'ext_target': ext_t,
                'warnings': warnings}

    def set_height(self, target_fixed, lift_cmd):
        """Closed-loop lift: returns the final lift command.

        move_joints can return while the joint is still short, and the lift
        settles ~1 cm below its command, so the measured error is ADDED to the
        command (re-sending the same target would not help).
        """
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.move_joints(['joint_lift'], [lift_cmd], secs=g('approach_secs'))
        for attempt in range(3):
            time.sleep(0.5)
            err = np.array(self.plan(target_fixed)['goal_base']) - self.grasp_in_base()
            self.get_logger().info(f'Grasp-centre height error (m): {err[2]:+.3f}')
            if abs(err[2]) <= g('z_tol'):
                return lift_cmd
            if attempt == 2:
                raise RuntimeError(f'grasp centre still {err[2]:+.3f} m off in z - '
                                   'lift blocked? (not closing the gripper)')
            lift_cmd = min(max(lift_cmd + float(err[2]), LIFT_RANGE[0]), LIFT_RANGE[1])
            self.move_joints(['joint_lift'], [lift_cmd])

    def extend_to(self, target_fixed):
        """Extend until the grasp centre is at the block along the arm axis."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        for attempt in range(2):
            p = self.plan(target_fixed)
            err = float(self.arm_axis @ (np.array(p['goal_base']) - self.grasp_in_base()))
            if abs(err) <= g('y_tol'):
                return
            if attempt == 1:
                self.get_logger().warn(f'extension still {err:+.3f} m off - closing anyway')
                return
            self.move_joints(['wrist_extension'], [p['ext_target']],
                             secs=g('approach_secs'))

    def run(self, target_fixed):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            t0 = time.time()
            while not self.joints and time.time() - t0 < 5.0:
                time.sleep(0.1)
            if not self.joints:
                raise RuntimeError('no /stretch/joint_states yet')
            # pregrasp just finished: don't plan from TF older than its final pose
            self.mark_settled()
            self.get_logger().info(f'Target in {self.fixed}: {np.round(target_fixed, 3)}')
            p = self.plan(target_fixed)
            self.get_logger().info(f'Grasp plan: {p}')
            if not self.execute:
                self.get_logger().info('DRY RUN - rerun with -p execute:=true to move.')
                return
            if p['warnings']:
                raise RuntimeError('; '.join(p['warnings']))
            if not self.client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('trajectory action server not available')

            # 1. open + orient (wrist changes move the grasp centre -> replan after)
            self.move_joints([GRIPPER], [g('gripper_open')])
            self.move_joints(['joint_wrist_pitch', 'joint_wrist_yaw'],
                             [g('wrist_pitch'), g('wrist_yaw')])
            time.sleep(0.5)
            p = self.plan(target_fixed)
            if self.approach == 'side':
                # 2. grasp height while still short of the block, 3. straight in
                lift_cmd = self.set_height(target_fixed, p['lift_target'])
                self.extend_to(target_fixed)
            else:
                # 2. over the block at hover height, 3. straight down
                self.extend_to(target_fixed)
                lift_cmd = self.set_height(target_fixed, p['lift_target'])
            err = np.array(self.plan(target_fixed)['goal_base']) - self.grasp_in_base()
            self.get_logger().info(f'Grasp-centre error before closing (m): {np.round(err, 3)}')
            # 4. close
            self.move_joints([GRIPPER], [g('gripper_close')])
            time.sleep(1.0)
            closed = self.joints.get(GRIPPER)
            # 5. lift
            self.move_joints(['joint_lift'],
                             [min(lift_cmd + g('lift_height'), LIFT_RANGE[1])])
            if closed is not None and closed <= g('gripper_close') + 0.05:
                self.get_logger().warn(
                    f'Gripper closed fully ({closed:.3f}) - it probably missed the block.')
            else:
                self.get_logger().info(f'DONE. Block grasped and lifted (gripper {closed}).')
        except Exception as e:
            self.get_logger().error(f'Grasp aborted: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Grasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
