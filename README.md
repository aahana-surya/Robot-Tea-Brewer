# Robot-Tea-Brewer
Teaching the Stretch Robot to make your tea!

# Stretch Commands

Always build and source from the workspace root (`~/ros2_ws`), never from `src/` or this
repo folder - that creates a second, stale `install/` that's easy to source by mistake.
```bash
cd ~/ros2_ws
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
```bash
stretch_robot_home.py
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
ros2 launch stretch_core stretch_realsense.launch.py   # head camera (D435i)
ros2 launch stretch_core d405_basic.launch.py          # gripper camera (D405)
rviz2 (see the camera images being published)
```

### Yolo world: pregrasp + grasp

Parameters that differ between the simulator and the real robot:

| Parameter | Node(s) | Sim | Real robot | Why |
|---|---|---|---|---|
| `platform` | `yolo_world_localizer` | `sim` (default) | **`robot`** | Robot uses depth *aligned* to the color image. Without it the depth pixels don't line up with the color pixels: near a table edge the handle is measured on the floor behind it and the base drives far too far. |
| `use_sim_time` | `pregrasp`, `grasp`, `steep_tea` | `true` (default) | **`false`** | The real robot has no `/clock`; with sim time on, ROS time stays at 0 and timeouts never fire. |

Two localizer instances run at once: the head camera finds the handle from afar, and
the gripper camera (namespace `/gripper`, topic `/gripper/orange_block/position`)
re-measures it once pregrasp has the gripper above it. Keep `-r __ns:=/gripper`, or the
gripper instance publishes on the head instance's topic.

#### Simulator
Terminal 1 (simulation):
```bash
ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true mode:=navigation use_mujoco_viewer:=false use_rviz:=false use_robocasa:=false
```
Terminal 2 (yolo world, head camera):
```bash
ros2 run robot_tea_brewer yolo_world_localizer
```
Terminal 3 (yolo world, gripper camera):
```bash
ros2 run robot_tea_brewer yolo_world_localizer --ros-args -r __ns:=/gripper -p camera:=gripper
```
Terminal 4 (head scan, optional - looks for the orange block):
```bash
ros2 run robot_tea_brewer head_scan
```
Terminal 5 (drive/extend to the hover pose above the orange block, then re-measure
with the gripper camera and correct):
```bash
ros2 run robot_tea_brewer pregrasp
```
Terminal 6 (pick up the block - waits for pregrasp to publish `pregrasp/target`):
```bash
ros2 run robot_tea_brewer grasp
```

#### Real robot
Start the driver and both cameras first (see "Get the physical robot to run" above).

Terminal 1 (yolo world, head camera):
```bash
ros2 run robot_tea_brewer yolo_world_localizer --ros-args -p platform:=robot -p show:=true
```
Terminal 2 (yolo world, gripper camera):
```bash
ros2 run robot_tea_brewer yolo_world_localizer --ros-args -r __ns:=/gripper -p camera:=gripper -p platform:=robot -p show:=true
```
Terminal 3 (head scan, optional):
```bash
ros2 run robot_tea_brewer head_scan
```
Terminal 4 (pregrasp):
```bash
ros2 run robot_tea_brewer pregrasp --ros-args -p use_sim_time:=false
```
Terminal 5 (grasp):
```bash
ros2 run robot_tea_brewer grasp --ros-args -p use_sim_time:=false
```

#### Viewing the cameras
```bash
ros2 run rqt_image_view rqt_image_view /yolo_world/annotated_image           # head detections
ros2 run rqt_image_view rqt_image_view /gripper/yolo_world/annotated_image   # gripper detections
# raw gripper feed (real robot): /gripper_camera/color/image_rect_raw
```
Or add `-p show:=true` to a localizer to open an OpenCV window with its detections.

Check what each camera reports for the handle (base_link, metres; z should be about
table height, not ~0):
```bash
ros2 topic echo /orange_block/position
ros2 topic echo /gripper/orange_block/position
```

#### Useful parameters
`yolo_world_localizer`:
- `camera` - `head` (default) or `gripper`.
- `show` - `true` opens an OpenCV window with the detections.
- `period` - seconds between detections (default 2.0); raise it for the gripper
  instance (e.g. `3.0`) if the robot's CPU struggles with two models.
- `orange_hsv_lo` / `orange_hsv_hi` - orange color thresholds (tuned on the sim);
  relax e.g. `-p orange_hsv_lo:="[5, 120, 100]"` if the handle is missed in real lighting.

`pregrasp`:
- `execute` - `false` prints the plan without moving.
- `approach` - `top` (default, gripper pointing down) or `side`.
- `hover` - m above the handle for the top approach (default 0.10).
- `refine` - re-measure with the gripper camera (default `true`); `false` skips it.
- `refine_samples` / `refine_timeout` - gripper detections to average (default 5) and
  s to wait for them (default 30). On timeout the head-camera position is kept.
- `max_correction` - m; a gripper-camera position further than this from the head one
  is treated as a wrong detection and ignored (default 0.08).
- `x_tol`, `drive_tries` - base alignment tolerance (m, default 0.01) and correction
  rounds (default 3).
- `base_speed` / `min_speed` - top base speed and the slowest speed near the goal
  (m/s, defaults 0.3 / 0.01).

`grasp`:
- `execute` - `false` prints the plan without moving.

### Steep the tea bag (bag already in the gripper)
After the yolo_world terminals above (the cup must be visible; use head_scan with
`--ros-args -p target_topic:=cup/position` if needed):
```bash
ros2 run robot_tea_brewer steep_tea --ros-args -p dips:=3 -p dip_depth:=0.12 -p bag_drop:=0.12
# dry run: -p execute:=false
# real robot: add -p use_sim_time:=false
```
`steep_tea` first goes above the cup using the head-camera position, then (`refine`,
default on) re-measures the cup with the gripper camera and moves again. This needs a
second localizer in another terminal:
```bash
ros2 run robot_tea_brewer yolo_world_localizer --ros-args -r __ns:=/gripper -p camera:=gripper
```
If the gripper camera sees nothing within `refine_timeout`, or its position differs from
the head estimate by more than `max_correction` (0.08 m), the head position is kept.
Disable with `-p refine:=false`.
The wrist orientation is no longer set here: `grasp` straightens it and rolls it `carry_roll_delta` (-270 deg, the same pose as +90 reached the other way round)
from the pregrasp roll after lifting the bag (`-p carry_wrist:=false` to skip, `carry_pitch`,
`carry_roll_delta` to change), and `steep_tea` keeps it as it finds it.

### Code layout
`stretch_motion.py` holds the shared motion code (`StretchMotion`, `TargetSampler`,
`plan_hover`); `pregrasp.py`, `grasp.py` and `steep_tea.py` are thin scripts on top of it.

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