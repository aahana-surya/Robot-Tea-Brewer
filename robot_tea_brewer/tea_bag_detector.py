"""ROS 2 node: detect a tea bag in a camera image using YOLO-World.

Reference implementation (restored). Not used by the current tag-based
pipeline. Needs: pip install ultralytics, plus CLIP (see notes in chat).
Prompt 'white paper tea bag' worked well in the sim; plain 'tea bag' did not.
"""
import json

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLOWorld

from robot_tea_brewer.color_detect import detect_tea_bag


class TeaBagDetector(Node):
    def __init__(self):
        super().__init__('tea_bag_detector')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('model', 'yolov8s-worldv2.pt')
        self.declare_parameter('prompts', ['white paper tea bag'])
        self.declare_parameter('conf', 0.05)  # sim renders: keep low
        self.declare_parameter('device', '')  # '' = auto, or 'cpu' / 'cuda:0'
        self.declare_parameter('show', False)
        self.declare_parameter('color_fallback', True)  # HSV if YOLO finds nothing
        self.declare_parameter('rotate', 0)  # 0/90/180/270 deg clockwise

        p = self.get_parameter
        self.conf = p('conf').value
        self.device = p('device').value or None
        self.show = p('show').value
        self.color_fallback = p('color_fallback').value
        self.rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(p('rotate').value)
        self.prompts = list(p('prompts').value)

        self.model = YOLOWorld(p('model').value)
        self.model.set_classes(self.prompts)  # open-vocabulary prompt(s)
        self.bridge = CvBridge()

        self.create_subscription(
            Image, p('image_topic').value, self.on_image, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(Image, 'tea_bag/annotated_image', 1)
        self.pub_pt = self.create_publisher(PointStamped, 'tea_bag/pixel', 10)
        self.pub_json = self.create_publisher(String, 'tea_bag/detections', 10)
        self.get_logger().info(
            f'Detecting {self.prompts} on {p("image_topic").value}')

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.rotate is not None:
            frame = cv2.rotate(frame, self.rotate)  # pixels are in rotated frame
        res = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False)[0]

        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                'label': self.prompts[int(box.cls)],
                'conf': float(box.conf),
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            })
        dets.sort(key=lambda d: d['conf'], reverse=True)

        annotated = res.plot()
        if not dets and self.color_fallback:
            dets = detect_tea_bag(frame)
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d['bbox'])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{d['label']} {d['conf']:.2f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        self.pub_json.publish(String(data=json.dumps(dets)))
        if dets:
            best = dets[0]
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y = best['center']
            pt.point.z = best['conf']  # confidence, not depth
            self.pub_pt.publish(pt)

        self.pub_img.publish(
            self.bridge.cv2_to_imgmsg(annotated, 'bgr8', header=msg.header))
        if self.show:
            cv2.imshow('tea_bag_detector', annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TeaBagDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
