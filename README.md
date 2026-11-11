# Robot-Tea-Brewer
Teaching the Stretch Robot to make your tea!

## YOLO detection (basic)

Runs YOLOv8 on a Stretch MuJoCo camera stream, publishes the annotated image on `/yolo/image`, and logs detected classes.

Prerequisite:

```bash
pip install ultralytics
```

Run:

```bash
# 1. Start the sim with cameras enabled
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true mode:=navigation
# 2. Run the basic
ros2 run robot_tea_brewer yolo_basic
```

Weights (`*.pt`) are git-ignored. Ultralytics downloads `yolov8n.pt` on first run, or point at a local copy in `weights/`:

```bash
ros2 run robot_tea_brewer yolo_basic --ros-args -p model:=$HOME/ros2_ws/src/Robot-Tea-Brewer/weights/yolov8s.pt
```
