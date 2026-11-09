"""ROS 2 node: locate the tea-bag tag in 3D (color detection + depth + TF).

Publishes geometry_msgs/PointStamped on `tea_tag/position` in `target_frame`
(default base_link), plus an annotated image on `tea_tag/annotated_image`.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from robot_tea_brewer.color_detect import detect_tag


def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel (u, v) at depth z [m] -> 3D point in the camera optical frame."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def depth_at(depth, u, v, half=2):
    """Median valid depth [m] in a small patch; None if no valid pixels."""
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    patch = depth[max(v - half, 0):min(v + half + 1, h),
                  max(u - half, 0):min(u + half + 1, w)].astype(np.float64)
    if depth.dtype == np.uint16:      # millimetres
        patch = patch / 1000.0
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.median(patch)) if patch.size else None


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class TagLocalizer(Node):
    def __init__(self):
        super().__init__('tag_localizer')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('info_topic', '/camera/depth/camera_info')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('show', False)
        p = self.get_parameter
        self.target = p('target_frame').value
        self.show = p('show').value

        self.bridge = CvBridge()
        self.info = None
        self.depth = None
        self.depth_frame = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, p('info_topic').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic').value,
                                 self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, p('color_topic').value,
                                 self.on_color, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointStamped, 'tea_tag/position', 10)
        self.pub_img = self.create_publisher(Image, 'tea_tag/annotated_image', 1)
        self.get_logger().info(
            f'Localizing green tag; output in {self.target} on tea_tag/position')

    def on_info(self, msg):
        self.info = msg

    def on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame = msg.header.frame_id

    def on_color(self, msg):
        if self.info is None or self.depth is None:
            self.get_logger().info('Waiting for camera_info and depth...',
                                   throttle_duration_sec=3.0)
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        tag = detect_tag(frame)
        annotated = frame.copy()
        if tag is None:
            self.get_logger().info('Tag not visible', throttle_duration_sec=2.0)
            self.publish_img(annotated, msg.header)
            return

        # color pixel -> depth pixel (identical if the streams are aligned)
        u, v = tag['center']
        dh, dw = self.depth.shape[:2]
        du, dv = u * dw / frame.shape[1], v * dh / frame.shape[0]
        z = depth_at(self.depth, du, dv)
        x1, y1, x2, y2 = (int(c) for c in tag['bbox'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if z is None:
            self.get_logger().warn('No valid depth at tag pixel',
                                   throttle_duration_sec=2.0)
            self.publish_img(annotated, msg.header)
            return

        K = self.info.k  # row-major 3x3, intrinsics of the depth image
        p_cam = deproject(du, dv, z, K[0], K[4], K[2], K[5])
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target, self.depth_frame, Time(), Duration(seconds=0.5))
        except Exception as e:  # tf2 raises several exception types
            self.get_logger().warn(f'TF lookup failed: {e}',
                                   throttle_duration_sec=2.0)
            self.publish_img(annotated, msg.header)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        p_out = quat_to_matrix(q.x, q.y, q.z, q.w) @ p_cam + [t.x, t.y, t.z]

        out = PointStamped()
        out.header.frame_id = self.target
        out.header.stamp = msg.header.stamp
        out.point.x, out.point.y, out.point.z = (float(c) for c in p_out)
        self.pub.publish(out)
        cv2.putText(annotated, f'{self.target}: {p_out[0]:.3f} {p_out[1]:.3f} {p_out[2]:.3f}',
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        self.get_logger().info(
            f'tag @ {self.target}: x={p_out[0]:.3f} y={p_out[1]:.3f} z={p_out[2]:.3f} '
            f'(depth {z:.3f} m)', throttle_duration_sec=1.0)
        self.publish_img(annotated, msg.header)

    def publish_img(self, img, header):
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, header=header))
        if self.show:
            cv2.imshow('tag_localizer', img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TagLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
