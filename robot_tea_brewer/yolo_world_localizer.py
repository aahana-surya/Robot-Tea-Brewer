"""ROS 2 node: open-vocabulary 3D localization of the orange block and cup.

YOLO-World finds the targets from TEXT prompts only (no color thresholds, no
training). The best box per target is turned into a 3D point with the depth
image + camera intrinsics, then transformed with TF into `target_frame`.

Targets
  block : bright orange 3D-printed tea-bag handle, lying flat on the table.
          The handle is tiny in the head camera (~25 x 10 px), so when YOLO-World
          finds no box the largest bright-orange blob is used instead
          (`color_fallback`). Inside the box the orange pixels (HSV threshold) pick
          the depth sample, so it lands on the handle and not the table behind
          it. The head camera sees its top face, so the point is lowered by
          `block_drop` (half the thickness) to the handle's centre.
          Published on `orange_block/position`.
          (Upright block instead: -p block_drop:=0.0 -p block_half_depth:=<half width>.)
  cup   : published on `cup/position` (centre of the cup axis at the rim).

Also publishes annotated images on `yolo_world/annotated_image`.

Camera: `-p camera:=head` (default, D435i, mounted sideways) or `-p camera:=gripper`
(D405 on the wrist; the block is far bigger in this view once the arm is near it).
Platform: `-p platform:=sim` (default, stretch_mujoco topics) or `-p platform:=robot`
(realsense-ros topics, with depth ALIGNED to the color image - the raw depth
image has a different field of view, so colour pixels don't line up with it).
The presets only set the defaults of color_topic / depth_topic / info_topic / rotate;
any of those can still be overridden individually.

To run the head and gripper cameras at once (pregrasp refines with the gripper one),
start a second instance in its own namespace:
  ros2 run robot_tea_brewer yolo_world_localizer --ros-args -r __ns:=/gripper \
      -p camera:=gripper              # -> /gripper/orange_block/position
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

# (platform, camera) preset -> (color_topic, depth_topic, info_topic, rotate)
# head camera is mounted sideways (rotate 90)
CAMERAS = {
    ('sim', 'head'): ('/camera/color/image_raw', '/camera/depth/image_rect_raw',
                      '/camera/depth/camera_info', 90),
    ('sim', 'gripper'): ('/gripper_camera/image_raw', '/gripper_camera/depth/image_rect_raw',
                         '/gripper_camera/depth/camera_info', 0),
    # realsense-ros (stretch_core d435i_*/d405_basic launch, align_depth enabled)
    ('robot', 'head'): ('/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
                        '/camera/aligned_depth_to_color/camera_info', 90),
    ('robot', 'gripper'): ('/gripper_camera/color/image_rect_raw',
                           '/gripper_camera/aligned_depth_to_color/image_raw',
                           '/gripper_camera/aligned_depth_to_color/camera_info', 0),
}

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
        cam = d('camera', 'head').value
        platform = d('platform', 'sim').value
        if (platform, cam) not in CAMERAS:
            raise ValueError(f"(platform, camera) must be one of {list(CAMERAS)}, "
                             f"got {(platform, cam)!r}")
        color_t, depth_t, info_t, rot = CAMERAS[(platform, cam)]
        d('color_topic', color_t)
        d('depth_topic', depth_t)
        d('info_topic', info_t)
        d('target_frame', 'base_link')
        d('model', 'yolov8s-worldv2.pt')
        d('device', '')
        d('rotate', rot)
        d('period', 2.0)                      # s between inferences (CPU friendly)
        d('threads', 2)                       # torch CPU threads; keep few so the sim isn't starved
        d('imgsz', 640)                       # inference size; upscaling the 424x240 head cam further hurts
        d('block_prompt', 'orange I-shaped plastic handle')
        d('block_conf', 0.05)
        d('block_half_depth', 0.0)            # m, horizontal push visible face -> centre (upright objects)
        d('block_drop', 0.007)                # m, lower the top-face point to the centre (flat handle)
        # OpenCV HSV (H 0-179). S >= 170 keeps the orange-brown wood table out
        # (wood S ~80-100, handle S ~235 in the sim render)
        d('orange_hsv_lo', [5, 170, 150])
        d('orange_hsv_hi', [22, 255, 255])
        d('orange_min_px', 15)                # fewer orange pixels -> use the box centre
        # if YOLO finds no box, use the largest bright-orange blob in the whole image.
        # The handle is only ~25 x 10 px in the head camera - too small for YOLO-World.
        d('color_fallback', True)
        d('cup_prompt', 'paper cup')
        d('cup_conf', 0.25)
        d('cup_radius', 0.045)                # m, to move from visible wall to cup axis
        d('cup_half_height', 0.05)            # m, to get the rim from the box centre
        # 'robot' is left out on purpose: near the arm it out-scores the block and steals its boxes
        d('distractors', ['table', 'wall'])
        d('show', False)
        # Deproject in the COLOR image's frame. The pixels come from the color image
        # and the sim renders depth from the color camera's viewpoint (aligned), but
        # stamps it with the depth optical frame, ~1.3 cm away in TF. Using the depth
        # frame shifted every detection ~1.2 cm toward the robot and ~8 mm down.
        # Real robot: use an aligned depth topic (e.g. -p depth_topic:=
        # /camera/aligned_depth_to_color/image_raw) with this left True.
        d('use_color_frame', True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.target = g('target_frame')
        self.rotate = g('rotate')
        self.cv_rot = ROTATIONS[self.rotate]
        torch.set_num_threads(int(g('threads')))
        self.period, self.show = g('period'), g('show')
        self.cfg = {'block': (g('block_prompt'), g('block_conf')),
                    'cup': (g('cup_prompt'), g('cup_conf'))}
        self.cup_r, self.cup_hh = g('cup_radius'), g('cup_half_height')
        self.block_hd = g('block_half_depth')
        self.block_drop = g('block_drop')
        self.hsv_lo = np.array(g('orange_hsv_lo'), np.uint8)
        self.hsv_hi = np.array(g('orange_hsv_hi'), np.uint8)
        self.orange_min_px = g('orange_min_px')
        self.color_fallback = g('color_fallback')
        self.imgsz = int(g('imgsz'))

        self.classes = [self.cfg['block'][0], self.cfg['cup'][0]] + list(g('distractors'))
        self.model = YOLOWorld(g('model'))
        self.model.set_classes(self.classes)
        self.device = g('device') or None

        self.bridge = CvBridge()
        self.info = self.depth = self.depth_frame = self.color_frame = None
        self.use_color_frame = g('use_color_frame')
        self.last_t = 0.0
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, g('info_topic'), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, g('color_topic'), self.on_color, qos_profile_sensor_data)
        self.pub = {k: self.create_publisher(PointStamped, f'{n}/position', 10)
                    for k, n in (('block', 'orange_block'), ('cup', 'cup'))}
        self.pub_img = self.create_publisher(Image, 'yolo_world/annotated_image', 1)
        self.get_logger().info(f'Camera {cam!r} on {platform} ({g("color_topic")}, '
                               f'depth {g("depth_topic")}, rotate {self.rotate}); '
                               f'YOLO-World prompts: {self.classes}')

    def on_info(self, msg):
        self.info = msg

    def on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame = msg.header.frame_id

    def orange_mask(self, bgr):
        return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), self.hsv_lo, self.hsv_hi)

    def orange_blob(self, img):
        """Bounding box (x1, y1, x2, y2) of the largest orange blob, or None."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(self.orange_mask(img))
        if n < 2:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[i]
        if area < self.orange_min_px:
            return None
        return [float(x), float(y), float(x + w), float(y + h)]

    def orange_center(self, img, box):
        """Centroid (rotated frame) of the orange pixels inside box, or None."""
        x1, y1, x2, y2 = (int(round(c)) for c in box)
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            return None
        mask = self.orange_mask(crop)
        ys, xs = np.nonzero(mask)
        if xs.size < self.orange_min_px:
            return None
        return max(x1, 0) + float(np.median(xs)), max(y1, 0) + float(np.median(ys))

    def orange_depth(self, img, box, frame_wh):
        """Median depth [m] over the orange pixels inside box (rotated frame), or None.

        For a small object a square patch around the centre also catches the
        table; sampling only orange pixels keeps the depth on the object.
        """
        w, h = frame_wh
        x1, y1, x2, y2 = (int(round(c)) for c in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        ys, xs = np.nonzero(self.orange_mask(img[y1:y2, x1:x2]))
        if xs.size < self.orange_min_px:
            return None
        u, v = rotated_to_original(xs + x1, ys + y1, self.rotate, w, h)
        dh, dw = self.depth.shape[:2]
        du = np.clip(np.round(np.asarray(u) * dw / w).astype(int), 0, dw - 1)
        dv = np.clip(np.round(np.asarray(v) * dh / h).astype(int), 0, dh - 1)
        z = self.depth[dv, du].astype(np.float64)
        if self.depth.dtype == np.uint16:      # millimetres
            z = z / 1000.0
        z = z[np.isfinite(z) & (z > 0.05)]
        return float(np.median(z)) if z.size else None

    def box_depth(self, box_rot, frame_wh, frac=0.6):
        """Median valid depth [m] over the central `frac` of a rotated-frame box, or None."""
        w, h = frame_wh
        x1, y1, x2, y2 = box_rot
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * frac / 2, (y2 - y1) * frac / 2
        gx, gy = np.meshgrid(np.linspace(cx - hw, cx + hw, 15), np.linspace(cy - hh, cy + hh, 15))
        u, v = rotated_to_original(gx.ravel(), gy.ravel(), self.rotate, w, h)
        dh, dw = self.depth.shape[:2]
        du = np.clip(np.round(np.asarray(u) * dw / w).astype(int), 0, dw - 1)
        dv = np.clip(np.round(np.asarray(v) * dh / h).astype(int), 0, dh - 1)
        z = self.depth[dv, du].astype(np.float64)
        if self.depth.dtype == np.uint16:      # millimetres
            z = z / 1000.0
        z = z[np.isfinite(z) & (z > 0.05)]
        return float(np.median(z)) if z.size else None

    def locate(self, box_rot, frame_wh, extra_depth=0.0, uv_rot=None, horizontal=False,
               z_cam=None):
        """Rotated-frame bbox (or pixel uv_rot) -> 3D point in target_frame (or None).

        extra_depth moves the visible-surface point to the object's axis: along the
        camera ray by default, or (horizontal=True) along the ray's horizontal part
        in target_frame. Use horizontal for upright objects seen from above: the head
        camera looks steeply down, so a push along the ray would mostly lower the
        point instead of moving it to the far side's centre.
        """
        w, h = frame_wh
        x1, y1, x2, y2 = box_rot
        ur, vr = uv_rot if uv_rot is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
        u, v = rotated_to_original(ur, vr, self.rotate, w, h)
        dh, dw = self.depth.shape[:2]
        du, dv = u * dw / w, v * dh / h
        z = z_cam if z_cam is not None else depth_at(self.depth, du, dv, half=3)
        if z is None and z_cam is None:
            # the centre pixel can fall on a depth hole (e.g. the dark cup opening, or
            # too close for the sensor): use the median over the box's central area
            z = self.box_depth(box_rot, frame_wh)
        if z is None:
            return None
        if not horizontal:
            z += extra_depth                       # surface -> object axis
        K = self.info.k
        p_cam = deproject(du, dv, z, K[0], K[4], K[2], K[5])
        tf = self.tf_buffer.lookup_transform(
            self.target, (self.color_frame if self.use_color_frame and self.color_frame
                          else self.depth_frame), Time(), Duration(seconds=0.5))
        t, q = tf.transform.translation, tf.transform.rotation
        cam = np.array([t.x, t.y, t.z])
        p = quat_to_matrix(q.x, q.y, q.z, q.w) @ p_cam + cam
        if horizontal and extra_depth:
            d = p - cam
            d[2] = 0.0
            n = np.linalg.norm(d)
            if n > 1e-6:
                p = p + extra_depth * d / n
        return p

    def on_color(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_t < self.period:
            return
        if self.info is None or self.depth is None:
            self.get_logger().info('Waiting for camera_info and depth...',
                                   throttle_duration_sec=3.0)
            return
        self.last_t = now
        self.color_frame = msg.header.frame_id
        # convert to BGR whatever the source encoding (e.g. rgb8 from the robot camera)
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        img = cv2.rotate(frame, self.cv_rot) if self.cv_rot is not None else frame
        res = self.model.predict(img, conf=min(c for _, c in self.cfg.values()),
                                 imgsz=self.imgsz, device=self.device, verbose=False)[0]

        # frame is always BGR here (converted above), whatever the source encoding
        annotated = img.copy()
        for i, key in enumerate(('block', 'cup')):
            prompt, thr = self.cfg[key]
            boxes = [b for b in res.boxes if int(b.cls) == i and float(b.conf) >= thr]
            if boxes:
                b = max(boxes, key=lambda b: float(b.conf))
                box, src = b.xyxy[0].tolist(), f'conf={float(b.conf):.2f}'
            elif key == 'block' and self.color_fallback and \
                    (box := self.orange_blob(img)) is not None:
                src = 'color'
            else:
                self.get_logger().info(f'{prompt!r} not visible', throttle_duration_sec=3.0)
                continue
            x1, y1, x2, y2 = (int(c) for c in box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'{prompt} {src}', (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            uv = extra = None
            if key == 'block':
                uv, extra = self.orange_center(img, box), self.block_hd
                if uv is None:
                    self.get_logger().warn('no orange pixels in block box - using box centre',
                                           throttle_duration_sec=3.0)
                else:
                    cv2.circle(annotated, (int(uv[0]), int(uv[1])), 4, (0, 0, 255), -1)
            else:
                extra = self.cup_r
            try:
                z_cam = self.orange_depth(img, box, (w, h)) if key == 'block' else None
                p = self.locate(box, (w, h), extra, uv, horizontal=(key == 'block'),
                                z_cam=z_cam)
            except Exception as e:  # tf2 raises several exception types
                self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
                continue
            if p is None:
                self.get_logger().warn(f'no valid depth for {prompt!r}', throttle_duration_sec=2.0)
                continue
            if key == 'cup':
                p[2] += self.cup_hh                # box centre is mid-height; use rim
            else:
                p[2] -= self.block_drop            # top face -> centre of a flat object
            out = PointStamped()
            out.header.frame_id = self.target
            out.header.stamp = msg.header.stamp
            out.point.x, out.point.y, out.point.z = (float(c) for c in p)
            self.pub[key].publish(out)
            self.get_logger().info(
                f'{prompt!r} {src} @ {self.target}: '
                f'{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}', throttle_duration_sec=1.0)
        # build the message by hand: cv_bridge.cv2_to_imgmsg breaks with OpenCV 5 (renumbered
        # CV_* type constants -> KeyError), and an explicit bgr8 encoding lets rqt show colour
        annotated = np.ascontiguousarray(annotated)
        out_img = Image()
        out_img.header = msg.header
        out_img.height, out_img.width = annotated.shape[:2]
        out_img.encoding = 'bgr8'
        out_img.step = out_img.width * 3
        out_img.data = annotated.tobytes()
        self.pub_img.publish(out_img)
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
