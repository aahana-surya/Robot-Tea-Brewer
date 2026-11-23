"""ROS 2 node: move Stretch's gripper to a hover point above the tea-bag tag.

Reads the tag position from `tea_bag/position` (published by
yolo_world_localizer, base_link frame), then (see StretchMotion.go_above):
  1. points the wrist down (pitch -1.57, yaw 0),
  2. drives the base forward/back so the gripper is level with the tag in x,
  3. raises the lift to the hover height (clear of the table),
  4. extends the arm until the gripper is above the tag.

DRY RUN by default: prints the plan and does not move. Pass `-p execute:=true`.
Needs the base in navigation mode (the node calls /switch_to_navigation_mode).
"""
import numpy as np
import rclpy
from rclpy.node import Node

from robot_tea_brewer.stretch_motion import (
    StretchMotion, TargetSampler, declare_motion_parameters, plan_hover, sim_time_overrides)


class Pregrasp(Node):
    def __init__(self):
        super().__init__('pregrasp', parameter_overrides=sim_time_overrides())
        declare_motion_parameters(self)
        self.declare_parameter('tag_topic', 'tea_bag/position')
        self.declare_parameter('hover', 0.10)       # m above tag
        self.declare_parameter('samples', 10)
        self.declare_parameter('execute', True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.hover, self.execute = g('hover'), g('execute')

        self.motion = StretchMotion(self)
        self.sampler = TargetSampler(self, self.motion, g('tag_topic'), g('samples'), self.run)
        self.get_logger().info(
            f'Collecting {g("samples")} tag samples '
            f'({"EXECUTE" if self.execute else "DRY RUN"})...')

    def run(self, tag_fixed):
        m = self.motion
        try:
            self.get_logger().info(f'Tag in {m.fixed}: {np.round(tag_fixed, 3)}')
            if not self.execute:
                if not m.joints:
                    raise RuntimeError('no /stretch/joint_states yet')
                p = plan_hover(m.in_base(tag_fixed), m.grasp_in_base(),
                               m.joints['joint_lift'], m.joints['wrist_extension'],
                               self.hover, m.arm_axis)
                self.get_logger().info(f'Plan (current wrist pose): {p}')
                self.get_logger().info('DRY RUN - rerun with -p execute:=true to move.')
                return
            err = m.go_above(tag_fixed, self.hover)
            self.get_logger().info(f'DONE. Gripper-to-hover error in base_link (m): {np.round(err, 3)}')
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
