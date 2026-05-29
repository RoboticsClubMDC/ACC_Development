# MDC North AI and Robotics ACC2026 
Repository for Quanser ACC competition 2026



# Autonomous QCar2 -- Node Documentation

---

## nav_to_pose.py

The core navigation controller for the QCar2. The node runs a timer at 80 Hz that handles everything: pose estimation, waypoint tracking and steering output all in the same loop.

**Pose estimation** is done with two filters running in parallel. The main one is an Extended Kalman Filter with three states: x position, y position and heading. The prediction step uses a bicycle kinematic model where the input is measured wheel speed and current steering angle. Every time a valid TF transform comes through from the localisation stack it runs a correction step using the x, y and yaw from that transform. The second filter is a simpler linear Kalman Filter just for heading. It takes the raw gyro z-rate as its process input and the TF yaw as its measurement. The output of that heading KF feeds into the main EKF correction rather than using the raw TF yaw directly, which smooths out any jitter in the heading estimate.

**Path following** uses pure pursuit. Waypoints come from `SDCSRoadMap` as a 2xN matrix in the QLabs world coordinate frame. Each tick the node grabs the current target waypoint and transforms it into the ROS map frame using a rotation matrix built from a tunable degree offset and a 2D translation offset. The pure pursuit steering angle is computed as `atan2(2 * L * sin(psi), d)` where L is the wheelbase, psi is the angle to the target waypoint in the car's local frame and d is the distance to it. That raw angle gets damped by subtracting a filtered gyro rate term multiplied by a derivative gain, which reduces oscillation when the car is correcting.

**Waypoint advancement** uses a speed-dependent lookahead. The car advances to the next waypoint when its distance to the current one drops below `v * 1.7` where v is the measured wheel speed. This means at higher speeds it looks further ahead and advances sooner, which keeps the path following smooth instead of the car chasing each waypoint to completion before moving to the next.

**Speed control** is mostly fixed at the `desired_speed` parameter but the node slows down in the final 100 waypoints and stops completely when within 0.25 m of the last waypoint in the path.

**TripPlanner integration** is handled through a `/cmd_waypoints` subscription. When the TripPlanner publishes a path there the node swaps out its entire waypoint array at runtime, resets the waypoint index and resumes following the new path. A flag tracks whether the current waypoints came from TripPlanner (already in ROS frame) or from SDCSRoadMap (needs the frame transform applied), so the transform doesn't get applied twice.

The node also publishes `/robot_pose` as a `PoseStamped` on every TF update so TripPlanner knows where the car is and can plan the next leg of the mission.

---

## yolo_detector.py

Traffic sign and traffic light detection node using Quanser's YOLOv8 wrapper paired with a depth-aligned RGB-D camera. It controls whether the car is allowed to move by publishing on `/motion_enable`.

**Camera input** comes from `QCar2DepthAligned` which provides a synchronised RGB image and depth map from the front-facing camera. Both frames are read on a 30 Hz timer.

**Inference** uses Quanser's `YOLOv8` class which wraps ultralytics. Detection runs on classes 9 (traffic light), 11 (stop sign) and 33 (yield sign) with a base confidence threshold of 0.3. Post-processing from Quanser's library estimates a distance for each detection using the aligned depth frame.

**Stop sign logic** requires confidence above 0.9 and a measured distance under 1.0 m. When triggered it sets `motion_enable` to false for 3 seconds. A 10 second cooldown prevents the same sign from triggering again immediately after the car moves.

**Yield sign logic** is the same as stop sign but with a 1.5 second stop duration.

**Traffic light logic** is different. The `lightColor` attribute from Quanser's post-processing classifies the light as red, yellow, green or idle. The node only acts on red or yellow with confidence above 0.5 and distance between 0.5 and 2.5 m. When triggered the stop window is only 0.25 seconds but `detection_cooldown` is set to zero which means the window refreshes every detection cycle. The car stays stopped for as long as the light remains red or yellow and starts moving as soon as the colour clears.

**Model management** handles the case where the weights file doesn't exist on disk. On startup if the file is missing or too small it downloads from a Quanser Box URL in 1 MB chunks to a temporary file and atomically renames it to the target path once complete.

---

## trip_planner.py

The high-level mission coordinator. It manages a taxi-style loop where the car travels from a home hub to a pickup location, waits, goes to a dropoff location, waits and returns to the hub. It doesn't do any driving itself, it just tells `nav_to_pose` where to go and tracks whether it got there.

**State machine** has six states: IDLE, TO_PICKUP, WAIT_AT_PICKUP, TO_DROPOFF, WAIT_AT_DROPOFF and TO_HUB. Transitions happen either on a path completion event from `/path_status` or when a wall-clock timer expires for the wait states.

**Path planning** works by finding the closest roadmap node to the car's current position and the closest roadmap node to the goal coordinate, then calling SDCSRoadMap to get the waypoint array between them. The exact goal coordinate gets appended as the final column of the waypoint array so the car actually drives to the target position rather than stopping at the nearest node.

**Coordinate conversion** is needed because SDCSRoadMap works in the QLabs world frame and nav_to_pose expects waypoints in the ROS map frame. The conversion applies a 2D rotation matrix built from `rotation_offset` degrees and then adds the `translation_offset` vector. The same parameters are used in nav_to_pose for consistency.

**Snap-to-exact** runs after the main path completes. It computes a two-point path from the car's current position directly to the goal coordinate and sends that. This corrects the residual gap between where the roadmap path ends and where the car actually needs to be.

**Live dispatch** works through ROS 2 parameter callbacks. Pickup and dropoff coordinates are declared as parameters. When they change at runtime and the planner is currently idle it sets `new_ride_requested` which kicks off a new mission on the next loop tick. This means a new trip can be dispatched by running a single `ros2 param set` command.

**LED control** sends parameter updates to `/qcar2_hardware` to change the car's LED colour at each mission stage. Green means heading to pickup, blue means passenger on board, orange means heading back, magenta means idle at hub. The hardware client is optional and the node continues normally if the service isn't available.
