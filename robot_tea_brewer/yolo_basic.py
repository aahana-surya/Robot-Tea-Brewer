#!/usr/bin/env python3
"""
YOLO object detection for the Stretch MuJoCo simulator (ROS 2 Humble).

1. Start the sim with cameras enabled:
     ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true mode:=navigation
2. Run this node:
     python3 yolo_basic.py
     python3 yolo_basic.py --ros-args -p image_topic:=/gripper_camera/image_raw -p show:=false

Publishes the annotated image on /yolo/image and logs detected classes.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

# The sim publishes the head (D435i) and nav cameras in their mounted, sideways
# orientation. YOLO does much better on upright images, so rotate before inference.
ROTATIONS = {
    "/camera/color/image_raw": cv2.ROTATE_90_CLOCKWISE,
    "/navigation_camera/image_raw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class YoloDemo(Node):
    def __init__(self):
        super().__init__("yolo_demo")
        self.image_topic = self.declare_parameter("image_topic", "/camera/color/image_raw").value
        model_path = self.declare_parameter("model", "yolov8n.pt").value
        self.conf = self.declare_parameter("conf", 0.4).value
        self.show = self.declare_parameter("show", True).value

        self.model = YOLO(model_path)  # downloads weights on first run
        self.bridge = CvBridge()

        # The sim driver publishes images as BEST_EFFORT; a RELIABLE subscriber would get nothing.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.pub = self.create_publisher(Image, "/yolo/image", 1)
        self.get_logger().info(f"Running {model_path} on {self.image_topic}")

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rotation = ROTATIONS.get(self.image_topic)
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
        frame = np.ascontiguousarray(frame)

        result = self.model(frame, conf=self.conf, verbose=False)[0]
        annotated = result.plot()

        names = [f"{result.names[int(c)]} {s:.2f}" for c, s in zip(result.boxes.cls, result.boxes.conf)]
        if names:
            self.get_logger().info(", ".join(names), throttle_duration_sec=1.0)

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)

        if self.show:
            cv2.imshow("YOLO", annotated)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
