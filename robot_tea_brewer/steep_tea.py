"""ROS 2 node: carry the (already gripped) tea bag over the cup and steep it.

Reads the cup position from `cup/position` (yolo_world_localizer, centre of the
cup axis at the rim), then:
  1. goes above the cup (StretchMotion.go_above), with the lift raised before
     the base moves so the bag clears the table,
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
        d('clearance', 0.05)      # m, bag bottom above the rim while travelling
        d('dip_depth', 0.12)      # m the lift lowers per dip
        d('dips', 3)
        d('dip_time', 1.5)        # s per down / up move
        d('dwell', 1.0)           # s at the bottom of each dip
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.execute = g('execute')
        self.hover = g('bag_drop') + g('clearance')
        self.dip_depth, self.dips = g('dip_depth'), g('dips')
        self.dip_time, self.dwell = g('dip_time'), g('dwell')

        self.motion = StretchMotion(self)
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

    def run(self, cup_fixed):
        m = self.motion
        try:
            self.get_logger().info(f'Cup rim in {m.fixed}: {np.round(cup_fixed, 3)}')
            if not self.execute:
                self.get_logger().info(
                    f'DRY RUN - would hover {self.hover:.3f} m over the rim and dip '
                    f'{self.dips}x by {self.dip_depth} m. Use -p execute:=true to move.')
                return
            m.go_above(cup_fixed, self.hover, raise_before_drive=True)
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
