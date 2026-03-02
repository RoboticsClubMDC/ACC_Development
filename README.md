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

## lane_stanley_node.py

A lane centering controller that computes steering corrections purely from the binary lane mask coming out of `lane_detection.py`. It deliberately avoids BEV or any homography and works entirely in raw image coordinates.

**Input** is the mono8 binary mask on `/lane_detection/lane_selected` where white pixels are the detected lane and everything else is black. The node also subscribes to `/qcar2_joint` to track measured wheel speed, although speed is not currently used in the steering calculation.

**Measurement band** is a horizontal crop of the mask between `band_y0_frac` and `band_y1_frac` of the image height, defaulting to 65 to 95 percent. This bottom portion of the image is where the lane is closest to the car and the perspective distortion is most consistent. Using the full image would include distant lane segments that move around a lot and would destabilise the centroid estimate.

**Cross-track error** is computed from the centroid x-coordinate of all white pixels in the band. The car center is assumed to be at the middle of the image width. The pixel offset of the centroid from center gets normalised by half the image width to produce a value in the range -1 to 1. Positive means the lane center is to the right of the car center in the image, meaning the car needs to steer right to center itself.

**Heading error** is estimated by dividing the band into N horizontal strips (default 10) and computing the centroid x for each strip that has enough white pixels. Those centroids at their respective row positions get fed into a first-degree polynomial fit: `x = m * y + b`. The slope m represents how much the lane is drifting laterally as you move up the image, which corresponds to the heading of the lane relative to the camera. That slope gets converted to an angle via `atan(m)` and normalised by 45 degrees to put it in a -1 to 1 range.

**Trust** is a binary 0 or 1 based on whether the total number of white pixels in the band exceeds `min_lane_px`. If the lane isn't clearly visible the node publishes zeros on all channels and resets the filter state so it doesn't carry stale values forward.

**EMA smoothing** runs on both the CTE and heading signals before they go into the steering law. The filter is `filtered = (1 - alpha) * filtered + alpha * new_value`. With alpha at 0.2 the filter strongly prefers the recent history over any single frame, which prevents the noisy mask edges from causing frame-to-frame steering jitter.

**Steering output** is a weighted sum: `delta = k_cte * cte + k_head * heading`, clipped to `max_steer`. The heading weight is higher than the CTE weight by default because heading error tends to build up faster than lateral offset and correcting it early prevents the CTE from growing.

---

## lane_keeping.py

A reactive safety guardrail that runs between the path planner output and the final actuator command. It never originates motion commands itself. It takes whatever the path planner sends and either passes it through or nudges the steering away from a detected sidewalk boundary.

**Data flow** is: `nav_to_pose` publishes to `/cmd_vel_raw`, this node reads that, modifies it if necessary and republishes on `/cmd_vel_nav` which goes to the actuator.

**Sidewalk sensing** reads the dilated no-go mask from `/sidewalk_detection/no_go_margin`. The node crops a near-field band from that mask between `band_y0_frac` and `band_y1_frac` of the image height, typically 55 to 95 percent. Within that band it computes overall occupancy (fraction of pixels that are white) and left vs right occupancy separately. The imbalance is `right_occ - left_occ` clamped to -1 to 1. A positive imbalance means more no-go pixels are on the right side so the car should steer left to move away.

**Repulsion bias** is scaled by two things: the imbalance and the overall occupancy. The occupancy scale is `min(occ * 5, 1.0)` which means at low overall occupancy (sparse noise blobs) the repulsion is weak even if the imbalance is high. As actual sidewalk fills more of the band the scaling increases toward full gain. The final bias is `gain * imbalance * occ_scale` clamped to `repulse_max`.

**Rate limiting** is applied to the bias state using a slew rate of `repulse_rate` radians per second. On each call the bias is only allowed to move toward the desired value by at most `rate * dt` per step. The final steering output goes through a second rate limiter as well. This is what prevents the YOLO mask's frame-by-frame variation from directly jerking the steering.

**Speed reduction** happens in two tiers. When occupancy exceeds `slow_occ` the speed is scaled down linearly. When it exceeds `panic_occ` the speed is hard capped at `panic_speed`. This handles cases where the car is heading directly into a sidewalk and the steering alone might not react fast enough.

**Mask timeout** means if no mask has arrived within `mask_timeout_sec` the node treats mask_ok as false and the bias stays at zero. The nav command passes through unmodified in this case.

---

## lane_detection.py

YOLO-based segmentation node that finds the drivable lane in each CSI camera frame and publishes a clean binary mask of the lane the car is currently on.

**Inference** runs `ultralytics` YOLO on every incoming frame. The model outputs masks and bounding boxes for each detected instance. Only class 0 (lane) detections are used. All class 0 masks get resized to the original image resolution and unioned together into a single boolean array. This handles cases where the model splits one continuous lane into multiple overlapping detections.

**Connected components** is then run on the union mask. This separates out distinct lane regions that aren't connected to each other. Any component with fewer than 200 pixels is discarded as noise. The remaining components are candidates for selection.

**Hysteresis tracking** picks the final lane from the candidates. On the first frame (or after tracking is lost) the target centroid is seeded at `lane_seed_x_frac * image_width`, defaulting to 55 percent across which puts it slightly right of center. On subsequent frames the target is the centroid of the previously selected component. Candidates are sorted by distance from the target centroid with a tiebreak on size, and the closest one wins. This keeps the node locked onto one lane as long as it stays visible even if other lanes enter the frame.

**Tracking loss** is counted in frames. If no valid component is found for more than `lane_lost_frames` consecutive frames the tracker resets back to the seed position. This prevents it from staying locked to a stale centroid after the lane has genuinely left the frame.

**Output** is two topics: a BGR overlay image with the selected lane drawn in green at 45 percent alpha for visualisation, and a mono8 binary mask at 0 or 255 that downstream nodes use for control.

---

## sidewalk_detection.py

YOLO segmentation node that identifies sidewalks and other no-go zones in the camera image. The output is a safety mask that `lane_keeping.py` uses to keep the car away from those areas.

**Inference** is the same ultralytics YOLO pipeline as lane_detection but targeting class 2 which is the no-go zone class. All class 2 detections get unioned into a single boolean mask at full image resolution.

**Dilation** is applied to the raw union mask before publishing. A morphological dilation with an elliptical kernel of radius `NO_GO_MARGIN_PX` (10 pixels) expands the detected region outward in all directions. The purpose is to create a safety buffer so that `lane_keeping.py` starts reacting to the boundary before the car physically reaches it rather than exactly at the detected edge.

**Output** is two topics: a BGR overlay with detected no-go zones shown in red at 45 percent alpha, and a mono8 binary mask of the dilated region on `/sidewalk_detection/no_go_margin` which is what `lane_keeping.py` subscribes to.

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

---

## stanley_live_plot.py

A live debugging visualiser that plots cross-track error and heading error from `lane_stanley_node.py` in real time. It is a standalone script that connects to the ROS network as a node and drives a matplotlib window.

**Architecture** splits the work across two threads. ROS spin runs in a daemon background thread and writes incoming Float32 messages into two pairs of deques: one pair for CTE timestamps and values and one pair for heading error timestamps and values. Each deque is bounded to a maximum length that covers the display window. The main thread runs the matplotlib animation loop and reads from those deques on each frame refresh.

**Timestamps** are derived from the ROS clock on each incoming message. The first message received sets the reference time and all subsequent timestamps are expressed as seconds elapsed since that reference. This means the x-axis always starts near zero even if the node was launched well after the system started.

**Plot layout** is two vertically stacked subplots sharing the same x-axis. The top plot shows CTE in normalised units with dashed threshold lines at +/- 0.3. The bottom shows heading error. Both use a dark theme. The x-axis window rolls forward in real time showing only the last 30 seconds of data.

**Thread safety** on the deques is handled with a single `threading.Lock` that both the ROS callbacks and the animation update function acquire before reading or writing.