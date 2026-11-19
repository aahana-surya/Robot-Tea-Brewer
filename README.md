# Robot-Tea-Brewer
Teaching the Stretch Robot to make your tea!

# Stretch Commands

```bash
colcon build --packages-select robot_tea_brewer
source install/setup.bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### Launch simulator
```bash
source ~/ament_ws/install/setup.bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation use_robocasa:=false
```

### For running YOLO:
```bash
ros2 run robot_tea_brewer tag_localizer –ros-args -p show:=true (to see OpenCV image of what it’s detecting)
ros2 run robot_tea_brewer head_scan (move camera to detect
ros2 run robot_tea_brewer pregrasp 
```

### Get the physical robot to run, in ament_ws
``` bash
stretch_robot_home.py
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
Ros2 launch stretch_core stretch_real_sense.launch.py
ros2 launch stretch
rviz2 (see the camera images being published)
```

### Yolo world with pregrasp
Terminal 1 (simulation):
```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true mode:=navigation use_mujoco_viewer:=false use_rviz:=false use_robocasa:=false
```
Terminal 2 (yolo world):
```bash
ros2 run robot_tea_brewer yolo_world_localizer
```
Terminal 3 (head scan, optional):
```bash
ros2 run robot_tea_brewer head_scan --ros-args -r tea_tag/position:=tea_bag/position
```

Terminal 4 (see camera feed, sim only):
```bash
ros2 run rqt_image_view rqt_image_view /yolo_world/annotated_image  
```

### Make the scene.xml:
```bash
python3 src/Robot-Tea-Brewer/tools/make_scene.py
```

# YOLO detection (basic)

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