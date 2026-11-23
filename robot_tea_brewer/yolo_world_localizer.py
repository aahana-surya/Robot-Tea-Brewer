"""ROS 2 node: open-vocabulary 3D localization of the tea bag and cup.

YOLO-World finds the targets from TEXT prompts only (no color thresholds, no
training). The best box per target is turned into a 3D point with the depth
image + camera intrinsics, then transformed with TF into `target_frame`.

Targets
  tea bag : we detect the orange 3D-printed handle attached to the tea bag
            string (the part the robot grips; detecting the handle is more
            reliable than detecting the tea bag itself). Published on
            `tea_bag/position`.
  cup     : published on `cup/position` (centre of the cup axis at the rim).

Also publishes annotated images on `yolo_world/annotated_image`.
"""
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLOWorld

from robot_tea_brewer.tag_localizer import deproject, depth_at, quat_to_matrix

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotated_to_original(u, v, rotate, w, h):
    """Map a pixel of the cv2.rotate()d frame back to the original (w x h) frame."""
    if rotate == 90:
        return v, h - 1 - u
    if rotate == 180:
        return w - 1 - u, h - 1 - v
    if rotate == 270:
        return w - 1 - v, u
    return u, v


class YoloWorldLocalizer(Node):
    def __init__(self):
        super().__init__('yolo_world_localizer')
        d = self.declare_parameter
        d('color_topic', '/camera/color/image_raw')
        d('depth_topic', '/camera/depth/image_rect_raw')
        d('info_topic', '/camera/depth/camera_info')
        d('target_frame', 'base_link')
        d('model', 'yolov8s-worldv2.pt')
        d('device', '')
        d('rotate', 90)                       # head camera is mounted sideways
        d('period', 2.0)                      # s between inferences (CPU friendly)
        d('threads', 2)                       # torch CPU threads; keep few so the sim isn't starved
        d('bag_prompt', 'orange plastic handle')
        d('bag_conf', 0.04)
        d('cup_prompt', 'paper cup')
        d('cup_conf', 0.25)
        d('cup_radius', 0.045)                # m, to move from visible wall to cup axis
        d('cup_half_height', 0.05)            # m, to get the rim from the box centre
        d('distractors', ['table', 'robot', 'wall'])
        d('show', False)
        d('color_encoding', 'bgr8')           # 'bgr8' (sim) or 'passthrough' (use camera's raw encoding)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.target = g('target_frame')
        self.rotate = g('rotate')
        self.cv_rot = ROTATIONS[self.rotate]
        torch.set_num_threads(int(g('threads')))
        self.period, self.show = g('period'), g('show')
        self.color_enc = g('color_encoding')
        self.cfg = {'bag': (g('bag_prompt'), g('bag_conf')),
                    'cup': (g('cup_prompt'), g('cup_conf'))}
        self.cup_r, self.cup_hh = g('cup_radius'), g('cup_half_height')

        self.classes = [self.cfg['bag'][0], self.cfg['cup'][0]] + list(g('distractors'))
        self.model = YOLOWorld(g('model'))
        self.model.set_classes(self.classes)
        self.device = g('device') or None

        self.bridge = CvBridge()
        self.info = self.depth = self.depth_frame = None
        self.last_t = 0.0
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, g('info_topic'), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, g('color_topic'), self.on_color, qos_profile_sensor_data)
        self.pub = {k: self.create_publisher(PointStamped, f'{n}/position', 10)
                    for k, n in (('bag', 'tea_bag'), ('cup', 'cup'))}
        self.pub_img = self.create_publisher(Image, 'yolo_world/annotated_image', 1)
        self.get_logger().info(f'YOLO-World prompts: {self.classes}')

    def on_info(self, msg):
        self.info = msg

    def on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame = msg.header.frame_id

    def locate(self, box_rot, frame_wh, extra_depth=0.0):
        """Rotated-frame bbox -> 3D point in target_frame (or None)."""
        w, h = frame_wh
        x1, y1, x2, y2 = box_rot
        u, v = rotated_to_original((x1 + x2) / 2, (y1 + y2) / 2, self.rotate, w, h)
        dh, dw = self.depth.shape[:2]
        du, dv = u * dw / w, v * dh / h
        z = depth_at(self.depth, du, dv, half=3)
        if z is None:
            return None
        z += extra_depth                           # surface -> object axis
        K = self.info.k
        p_cam = deproject(du, dv, z, K[0], K[4], K[2], K[5])
        tf = self.tf_buffer.lookup_transform(
            self.target, self.depth_frame, Time(), Duration(seconds=0.5))
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_matrix(q.x, q.y, q.z, q.w) @ p_cam + [t.x, t.y, t.z]

    def on_color(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_t < self.period:
            return
        if self.info is None or self.depth is None:
            self.get_logger().info('Waiting for camera_info and depth...',
                                   throttle_duration_sec=3.0)
            return
        self.last_t = now
        frame = self.bridge.imgmsg_to_cv2(msg, self.color_enc)
        out_enc = msg.encoding if self.color_enc == 'passthrough' else self.color_enc
        h, w = frame.shape[:2]
        img = cv2.rotate(frame, self.cv_rot) if self.cv_rot is not None else frame
        res = self.model.predict(img, conf=min(c for _, c in self.cfg.values()),
                                 device=self.device, verbose=False)[0]

        annotated = img.copy()
        for i, key in enumerate(('bag', 'cup')):
            prompt, thr = self.cfg[key]
            boxes = [b for b in res.boxes if int(b.cls) == i and float(b.conf) >= thr]
            if not boxes:
                self.get_logger().info(f'{prompt!r} not visible', throttle_duration_sec=3.0)
                continue
            b = max(boxes, key=lambda b: float(b.conf))
            box = b.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(c) for c in box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'{prompt} {float(b.conf):.2f}', (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            try:
                p = self.locate(box, (w, h), self.cup_r if key == 'cup' else 0.0)
            except Exception as e:  # tf2 raises several exception types
                self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
                continue
            if p is None:
                self.get_logger().warn(f'no valid depth for {prompt!r}', throttle_duration_sec=2.0)
                continue
            if key == 'cup':
                p[2] += self.cup_hh                # box centre is mid-height; use rim
            out = PointStamped()
            out.header.frame_id = self.target
            out.header.stamp = msg.header.stamp
            out.point.x, out.point.y, out.point.z = (float(c) for c in p)
            self.pub[key].publish(out)
            self.get_logger().info(
                f'{prompt!r} conf={float(b.conf):.2f} @ {self.target}: '
                f'{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}', throttle_duration_sec=1.0)
        # explicit encoding so rqt_image_view can display it (plain 8UC3 isn't a color format)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated, encoding=out_enc, header=msg.header))
        if self.show:
            cv2.imshow('yolo_world_localizer', annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
