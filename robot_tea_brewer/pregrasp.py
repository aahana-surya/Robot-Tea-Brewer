"""ROS 2 node: move Stretch's gripper to a pregrasp pose next to the target object.

Reads the target position from `orange_block/position` (published by
yolo_world_localizer, base_link frame; override with `-p tag_topic:=...`), then
(see StretchMotion.go_to):
  1. sets the wrist: pointing down and rolled 90 deg (pitch -1.57, roll 1.57,
     `approach:=top`, default - for the flat orange handle) or level
     (pitch 0, `approach:=side` - for an upright block),
  2. drives the base forward/back so the gripper is level with the target in x,
  3. moves the lift to the pregrasp height (arm still retracted),
  4. extends the arm until the gripper is `standoff` short of the object (side)
     or `hover` above it (top),
  5. (refine, default on) re-measures the object with the GRIPPER camera, which is
     now close to it and sees it much larger than the head camera did, then
     repeats 2-4 with that position. Needs a second localizer instance:
       ros2 run robot_tea_brewer yolo_world_localizer --ros-args \
           -r __ns:=/gripper -p camera:=gripper
     If the gripper camera reports nothing within `refine_timeout`, or a position
     more than `max_correction` from the head estimate (probably a wrong
     detection), the head-camera position is kept.

When done it publishes the target (fixed frame, latched) on `pregrasp/target`;
the `grasp` node waits for that message before picking the object up.

DRY RUN with `-p execute:=false`: prints the plan and does not move.
Needs the base in navigation mode (the node calls /switch_to_navigation_mode).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

from robot_tea_brewer.stretch_motion import (
    LATCHED, WRIST_DOWN, StretchMotion, TargetSampler, declare_motion_parameters,
    sim_time_overrides)


class Pregrasp(Node):
    def __init__(self):
        super().__init__('pregrasp', parameter_overrides=sim_time_overrides())
        declare_motion_parameters(self)
        d = self.declare_parameter
        # YOLO-World detection (yolo_world_localizer), base_link frame
        d('tag_topic', 'orange_block/position')
        # 'side': gripper level (pitch 0), stops `standoff` short of the object at
        #         about grasp height; `grasp` then extends straight in.
        # 'top' : gripper pointing down, `hover` above the object.
        d('approach', 'top')
        d('hover', 0.10)         # m above the object (top)
        d('wrist_roll', 1.57)    # top: fingers close along base y
        d('standoff', 0.08)      # m short of the object along the arm (side)
        d('side_z', 0.02)        # m above the detected centre (side)
        d('samples', 10)
        d('execute', True)
        d('refine', True)                       # re-measure with the gripper camera
        d('refine_topic', 'gripper/orange_block/position')
        d('refine_samples', 5)
        d('refine_timeout', 30.0)               # s (wall) to wait for the gripper samples
        d('max_correction', 0.08)               # m, larger -> distrust the gripper camera
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.execute, self.approach = g('execute'), g('approach')
        arm_axis = np.asarray(g('arm_axis'), float)
        if self.approach == 'side':
            self.offset = -g('standoff') * arm_axis + [0.0, 0.0, g('side_z')]
            self.wrist = (0.0, 0.0, 0.0)
        elif self.approach == 'top':
            self.offset = np.array([0.0, 0.0, g('hover')])
            self.wrist = (WRIST_DOWN, 0.0, g('wrist_roll'))
        else:
            raise ValueError(f"approach must be 'side' or 'top', got {self.approach!r}")

        self.motion = StretchMotion(self)
        self.target_pub = self.create_publisher(PointStamped, 'pregrasp/target', LATCHED)
        self.refine = g('refine')
        self.refine_timeout, self.max_corr = g('refine_timeout'), g('max_correction')
        self.refiner = (TargetSampler(self, self.motion, g('refine_topic'), g('refine_samples'))
                        if self.refine else None)
        self.sampler = TargetSampler(self, self.motion, g('tag_topic'), g('samples'), self.run)
        self.get_logger().info(
            f'Collecting {g("samples")} target samples '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    def refine_target(self, head_fixed, err):
        """Re-measure with the gripper camera and move again; returns (target, err)."""
        m = self.motion
        self.get_logger().info(
            f'Refining with the gripper camera ({self.refiner.topic}, '
            f'{self.refiner.n} samples)...')
        new = self.refiner.collect(timeout=self.refine_timeout)
        if new is None:
            self.get_logger().warn(
                'No gripper-camera position (is the /gripper localizer running and does '
                'it see the object?) - keeping the head-camera position.')
            return head_fixed, err
        corr = new - head_fixed
        self.get_logger().info(
            f'Gripper camera: {np.round(new, 3)} in {m.fixed}, correction '
            f'{np.round(corr, 3)} ({np.linalg.norm(corr) * 100:.1f} cm)')
        if np.linalg.norm(corr) > self.max_corr:
            self.get_logger().warn(
                f'Correction larger than max_correction={self.max_corr} m - probably a '
                'wrong detection; keeping the head-camera position.')
            return head_fixed, err
        err = m.go_to(new, self.offset, wrist=self.wrist)
        return new, err

    def run(self, tag_fixed):
        m = self.motion
        try:
            self.get_logger().info(f'Target in {m.fixed}: {np.round(tag_fixed, 3)}')
            m.wait_for_joints()
            if not self.execute:
                p = m.plan_to(tag_fixed, self.offset)
                self.get_logger().info(f'Plan (current wrist pose): {p}')
                self.get_logger().info('DRY RUN - rerun with -p execute:=true to move.')
                return
            # side approach lowers the lift, so pull the arm in first (a previous
            # run may have left it extended over the table)
            err = m.go_to(tag_fixed, self.offset, wrist=self.wrist,
                          retract_first=(self.approach == 'side'))
            if self.refine:
                tag_fixed, err = self.refine_target(tag_fixed, err)
            self.get_logger().info(
                f'DONE ({self.approach}). Gripper-to-pregrasp error in base_link (m): '
                f'{np.round(err, 3)}')
            out = PointStamped()
            out.header.frame_id = m.fixed
            out.header.stamp = self.get_clock().now().to_msg()
            out.point.x, out.point.y, out.point.z = (float(c) for c in tag_fixed)
            self.target_pub.publish(out)
            self.get_logger().info('Published pregrasp/target - grasp node may proceed.')
        except Exception as e:
            self.get_logger().error(f'Pregrasp aborted: {e}')
        finally:
            m.stop()


def main(args=None):
    rclpy.init(args=args)
    node = Pregrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.motion.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
