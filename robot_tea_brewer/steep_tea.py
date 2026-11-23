"""ROS 2 node: carry the (already gripped) tea bag over the cup and steep it.

Reads the cup position from `cup/position` (yolo_world_localizer, centre of the
cup axis at the rim), then:
  1. goes above the cup (StretchMotion.go_above), with the lift raised before
     the base moves so the bag clears the table (the wrist is left as `grasp`
     set it: straight and rolled, see its `carry_*` parameters),
  1b. (refine, default on) re-measures the cup with the GRIPPER camera, which is
     now close to it, and moves again with that position. Needs a second
     localizer instance:
       ros2 run robot_tea_brewer yolo_world_localizer --ros-args \
           -r __ns:=/gripper -p camera:=gripper
     If the gripper camera reports nothing within `refine_timeout`, or a position
     more than `max_correction` from the head estimate, the head-camera position
     is kept. (With a straight wrist the gripper camera may not see the cup; then the
     head position is used.)
  2. dips the bag into the cup and back out `dips` times (lift joint only),
     pausing `dwell` s at the bottom of each dip.

Height bookkeeping: the grasp center is placed `bag_drop + clearance` above the
rim, i.e. the BOTTOM of the hanging bag sits `clearance` above the rim. Each dip
lowers the lift by `dip_depth` (so the bag ends `dip_depth - clearance` below the
rim; keep that less than the cup depth).

DRY RUN with `-p execute:=false`. Needs the base in navigation mode.
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node

from robot_tea_brewer.stretch_motion import (
    LIFT_RANGE, StretchMotion, TargetSampler, declare_motion_parameters, sim_time_overrides)


class SteepTea(Node):
    def __init__(self):
        super().__init__('steep_tea', parameter_overrides=sim_time_overrides())
        declare_motion_parameters(self)
        d = self.declare_parameter
        d('cup_topic', 'cup/position')
        d('samples', 10)
        d('execute', True)
        d('bag_drop', 0.12)       # m from grasp center down to the bottom of the bag
        d('clearance', 0.10)      # m, bag bottom above the rim while travelling
        d('dip_depth', 0.07)      # m the lift lowers per dip
        d('dips', 3)
        d('dip_time', 1.5)        # s per down / up move
        d('dwell', 1.0)           # s at the bottom of each dip
        d('refine', True)                       # re-measure with the gripper camera
        d('refine_topic', 'gripper/cup/position')
        d('refine_samples', 5)
        d('refine_timeout', 30.0)               # s (wall) to wait for the gripper samples
        d('max_correction', 0.08)               # m, larger -> distrust the gripper camera
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.execute = g('execute')
        self.refine = g('refine')
        self.refine_timeout, self.max_corr = g('refine_timeout'), g('max_correction')
        self.hover = g('bag_drop') + g('clearance')
        self.dip_depth, self.dips = g('dip_depth'), g('dips')
        self.dip_time, self.dwell = g('dip_time'), g('dwell')

        self.motion = StretchMotion(self)
        self.refiner = (TargetSampler(self, self.motion, g('refine_topic'), g('refine_samples'))
                        if self.refine else None)
        self.sampler = TargetSampler(self, self.motion, g('cup_topic'), g('samples'), self.run)
        self.get_logger().info(
            f'Collecting {g("samples")} cup samples '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    def steep(self):
        m = self.motion
        top = m.joints['joint_lift']
        bottom = top - self.dip_depth
        if bottom < LIFT_RANGE[0]:
            raise RuntimeError(f'dip bottom {bottom:.3f} below lift range {LIFT_RANGE}')
        for i in range(self.dips):
            self.get_logger().info(f'Dip {i + 1}/{self.dips}')
            m.move_joints(['joint_lift'], [bottom], secs=self.dip_time)
            time.sleep(self.dwell)
            m.move_joints(['joint_lift'], [top], secs=self.dip_time)

    def go_above_cup(self, cup_fixed):
        # the wrist (straight, rolled) is set by `grasp`; keep it as it is
        j = self.motion.joints
        wrist = tuple(j[n] for n in ('joint_wrist_pitch', 'joint_wrist_yaw', 'joint_wrist_roll'))
        return self.motion.go_above(cup_fixed, self.hover, raise_before_drive=True, wrist=wrist)

    def refine_cup(self, head_fixed):
        """Re-measure with the gripper camera and move again; returns the cup position used."""
        m = self.motion
        self.get_logger().info(
            f'Refining with the gripper camera ({self.refiner.topic}, '
            f'{self.refiner.n} samples)...')
        new = self.refiner.collect(timeout=self.refine_timeout)
        if new is None:
            self.get_logger().warn(
                'No gripper-camera cup position (is the /gripper localizer running and '
                'does it see the cup?) - keeping the head-camera position.')
            return head_fixed
        corr = new - head_fixed
        self.get_logger().info(
            f'Gripper camera: {np.round(new, 3)} in {m.fixed}, correction '
            f'{np.round(corr, 3)} ({np.linalg.norm(corr) * 100:.1f} cm)')
        if np.linalg.norm(corr) > self.max_corr:
            self.get_logger().warn(
                f'Correction larger than max_correction={self.max_corr} m - probably a '
                'wrong detection; keeping the head-camera position.')
            return head_fixed
        self.go_above_cup(new)
        return new

    def run(self, cup_fixed):
        m = self.motion
        try:
            self.get_logger().info(f'Cup rim in {m.fixed}: {np.round(cup_fixed, 3)}')
            if not self.execute:
                self.get_logger().info(
                    f'DRY RUN - would hover {self.hover:.3f} m over the rim and dip '
                    f'{self.dips}x by {self.dip_depth} m. Use -p execute:=true to move.')
                return
            # wrist orientation is left as `grasp` set it (carry_wrist)
            m.wait_for_joints()
            self.go_above_cup(cup_fixed)
            if self.refine:
                self.refine_cup(cup_fixed)
            self.steep()
            self.get_logger().info('DONE steeping.')
        except Exception as e:
            self.get_logger().error(f'Steep aborted: {e}')
        finally:
            m.stop()


def main(args=None):
    rclpy.init(args=args)
    node = SteepTea()
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
