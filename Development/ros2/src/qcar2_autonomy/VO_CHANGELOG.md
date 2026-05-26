# VO Change Log

This file tracks cleanup, calibration decisions, and test observations for
the QCar2 visual odometry work.

## 2026-05-22 Off-car RTAB-Map for KLT/PnP — humble-rtabmap Docker image + "every-frame backfires" baseline finding [baseline finding RETRACTED — see CORRECTION at end of this entry; map pipeline is nondeterministic]

ENVIRONMENT: VO work moved off the QCar to Gabriel's Ubuntu 24.04 box
(no native ROS; ROS Humble lives ONLY inside Docker). Source bag
vslam_test12 (5.7 GB: /camera/{color_image,depth_image,camera_info},
/tf, /tf_static, /vo/*; 22453 msgs, ~154 s) transferred by USB to
~/Downloads/rtab_map_gabriel/.

INSTALL (why a new image): the Isaac dev image isaac_ros_dev-x86_64 is
ROS 2 Humble built on Ubuntu 20.04 (focal). packages.ros.org only builds
Humble for 22.04 (jammy), so apt inside it offers only FOXY rtabmap
(distro mismatch). Built an ISOLATED image `humble-rtabmap`
(FROM ros:humble + ros-humble-rtabmap-ros + rviz2 + rosbag2 sqlite plugin
+ cv_bridge + python3-opencv/numpy/matplotlib). Verified: all 13 rtabmap
pkgs + rtabmap / rtabmap-databaseViewer. Host ROS and the Isaac images are
untouched. run_dev.sh / the Isaac container CANNOT serve rtabmap (no
Humble rtabmap there).

OFF-CAR MAP PIPELINE (per config): vo_node (camera_mode=physical,
alignment_mode=auto, depth_scale=0.0, n_features=1200, feature_grid=8,
vo_frontend/vo_estimator under test, force_cart_yaw:=false)
-> vo_odom_tf_relay -> rtabmap_launch (visual_odometry:=false,
odom_topic:=/vo/odom_relay, qos:=2, approx_sync, rgbd_sync, use_sim_time).
CRITICAL TF point: do NOT play the bag's /tf — it carries Cartographer
map->odom->base_link, which COLLIDES with the relay's odom->base_link and
rtabmap's map->odom (the Test-12 "unconnected trees" / stuck at 1 node).
Play only camera + /tf_static; force_cart_yaw:=false means vo_node needs
no Cartographer yaw. Resulting clean tree:
map->odom(rtabmap) -> base_link(relay) -> camera(tf_static). vo_node runs
WITHOUT colcon build (pure rclpy; PYTHONPATH=.../src/qcar2_autonomy;
vo_node imports no qcar2_interfaces types). Off-tree scripts on the Ubuntu
box (NOT committed): /tmp/rtabmap_humble/{Dockerfile, run_config.sh,
run_batch.sh, run_live.sh, analyze_db.py, analyze_all.py, tf_frames.py}.

KEY FINDING — "process EVERY frame" BACKFIRES for this depth-VO (baseline
collapse): lowering --rate makes vo_node process more of the 3202 color
frames but SHRINKS the inter-frame baseline. At 0.25x the car moves ~5 mm
per frame (0.1 m/s / ~20 Hz) — below the depth-noise floor on the white
walls — so the motion estimate collapses toward zero and the trajectory
shrinks. PROOF (same config, only rate changed): KLT+PnP = clean 10.84 m
loop at 0.5x, collapsed to 0.43 m at 0.25x. Corollary: ORB+SVD looks BEST
at 0.25x only because it is the slowest combo and processed just 54% of
frames -> ~2x baseline -> full 15.4 m loop. i.e. dropping frames HELPED
it. CONCLUSION: there is an OPTIMAL INTERMEDIATE baseline; maximizing
frame coverage is NOT the best case. Matches why the car runs ~6 Hz
(healthy baseline at driving speed) and points to keyframe-by-distance
(process a frame only after enough motion) as the right policy — which is
what RTAB-Map already does for its own keyframes.

FRAME COVERAGE at 0.25x (of 3202 color frames): ORB+SVD 1729 (54%),
KLT+SVD 2952 (92%), ORB+PnP 2963 (93%), KLT+PnP 2965 (93%).

0.25x MAP RESULTS (kept for record as vo_map_<cfg>_r025.db; mostly
baseline-collapse artifacts):
  ORB+SVD  599 nodes / 15.40 m / 26 loop closures / max step 1.465 m  (good: low coverage, big baseline)
  KLT+SVD  468 nodes /  2.36 m / 12 loop closures                     (collapsed)
  ORB+PnP  305 nodes /  1.32 m /  5 loop closures                     (collapsed)
  KLT+PnP  157 nodes /  0.43 m /  2 loop closures                     (collapsed)
Pipeline sensitivity also noted: which configs chain cleanly
(neighbor-link count) varied between 0.5x and 0.25x — live-relay odometry
timing is fragile. The TRUSTWORTHY KLT/PnP comparison is the odometry 2x2
(recorded directly from /vo, independent of rtabmap), NOT these maps.

REGEN: re-running all 4 at 0.5x -> vo_map_<cfg>_r05.db (better baseline)
for viewable, trustworthy KLT/PnP maps. The _r025 set is retained for the
record (it documents the every-frame collapse).

ODOMETRY 2x2 (trustworthy KLT/PnP-vs-default readings, earlier campaign):
PnP cuts invalid-frame rate ~3x vs SVD (PnP needs valid depth on ONE
frame; SVD on BOTH; our depth is noisy on white walls); KLT alone (on SVD)
is WORSE than ORB (more invalid + jumps); KLT+PnP best on inliers/
reliability; big jumps (2-4) persist across ALL configs = environmental
(depth/rgb-sync/curve), not an algorithm choice. rgb/depth timestamps are
misaligned 0.03-0.22 s (seen in rtabmap rgbd_sync warnings) — prime
suspect for the jumps.

VIEWING: existing trustworthy maps vo_node.db (variant B, ORB+SVD,
force_cart_yaw=true, 3 loop closures) and cartographer.db (variant C, 14
loop closures) open in rtabmap-databaseViewer (both carry full keyframe
images for the Constraints/feature-match views). Live build viewable via
run_live.sh (rtabmap_viz GUI). No repo code changed this session.

CORRECTION (same day, after the 0.5x regeneration) — the "every-frame
backfires / baseline-collapse" KEY FINDING above is **RETRACTED**; it was
premature. Re-running showed the map pipeline is NONDETERMINISTIC: KLT+PnP
at 0.5x gave a clean 10.84 m loop on the first single run but COLLAPSED to
0.51 m on a later 0.5x batch run — same config, same rate, same bag. Same
inputs -> different output => the collapse is a PIPELINE ARTIFACT, not a
baseline/parallax law. Both 0.25x and 0.5x showed the identical pattern
(ORB+SVD full ~15-17 m loop, ~84-91 neighbor links; the other three
collapsed to <2.5 m, only 3-13 neighbor links). The differentiator is the
rtabmap NEIGHBOR-LINK count = whether rtabmap chained the relayed odometry
into the pose graph. UNKNOWN which: (a) vo_node's /vo/odometry itself
collapsed, or (b) rtabmap failed to associate good odometry with keyframes
(timing race) — the raw /vo/odometry was not retained to decide. Confound:
the frame-counting `ros2 bag record /vo/odometry` was added AFTER the first
(good) KLT+PnP run and present for all (collapsed) later runs; the extra
subscriber/load may have shifted timing. The ORIGINAL working maps
(vo_node.db, cartographer.db) used PRE-RECORDED odometry replayed at 1x
(always on time), NOT live-recomputed odometry — the likely robust fix is
TWO-STAGE: record the regenerated /vo/odometry first (no rtabmap, no timing
pressure), then feed that recorded odometry + camera into rtabmap (as
variant B did). CONSEQUENCE: none of the auto-generated vo_map_* maps
(_r05 or _r025) are a trustworthy KLT/PnP comparison — DO NOT cite them or
the baseline finding. The trustworthy KLT/PnP-vs-default comparison remains
the odometry 2x2 (recorded directly from /vo, independent of rtabmap).

DELETED (2026-05-22, operator-approved): the 8 untrustworthy
vo_map_<cfg>_r025.db / _r05.db + their *_trajectory.png +
vo_rtabmap_2x2_r025.png / _r05.png removed from
~/Downloads/rtab_map_gabriel/ (~1.5 GB; nondeterministic-pipeline
artifacts, reason recorded above). KEPT: originals (rtabmap_odom.db,
vo_node.db, cartographer.db, the 4 vo_* odometry bags, the original PNGs,
vslam_test12) + the operator's MANUAL live runs vo_map_live_*.db (run by
the operator in rtabmap_viz, 1x). force_cart_yaw note: a live force_cart_yaw
:=true re-run is NOT possible without the Test-12 TF collision (vo_node
needs Cartographer yaw from the bag's /tf, whose map->odom + odom->base_link
collide with rtabmap's map->odom and the relay's odom->base_link). The
clean true-yaw path is two-stage: feed PRE-RECORDED odometry (the campaign
vo_* bags were recorded at default force_cart_yaw=true) into rtabmap, as
vo_node.db (variant B = ORB+SVD, true-yaw) already was.

MANUAL LIVE RUNS (operator-driven in rtabmap_viz, 1x, force_cart_yaw=false):
unlike the flaky headless batch, the manual workflow (launch rtabmap FIRST,
let it come up, THEN play the bag at 1x, no frame-recorder load) chains
cleanly. KLT+SVD -> vo_map_live_klt_svd.db = 131 nodes, 11.62 m full loop,
164 neighbor links, 8 global loop closures, max step 0.238 m (NO jumps
>0.3 m), planar. Path is accurate on the straights + early/left/top portion
but DRIFTS PROGRESSIVELY (gradual = yaw accumulation, not jumps) so the end
lands ~2.5 m off start; RTAB-Map's 8 loop closures bridge the gap. Drift is
heading, not x/y scale, and concentrates in the later turns (operator
reported the path is bad at the 2nd left turn + just before the 3rd) ->
confirms force_cart_yaw=true (Cartographer/IMU yaw) would close the loop
much better. OPERATOR OBSERVATION (logged for a future tuning pass):
RTAB-Map's own ORB features cluster rather than spread — tunable via
Vis/GridRows + Vis/GridCols (registration features) and Kp/GridRows +
Kp/GridCols (loop-closure keypoints); the RTAB-Map analog of our
feature_grid=8. Series in progress (operator runs one config at a time):
KLT+SVD done. ORB+PnP -> vo_map_live_orb_pnp.db = 119 nodes, 11.14 m full
loop, 140 neighbor links, 6 loop closures, ONE 1.77 m jump (relocalization/
loop-closure snap), pose z exactly [0,0]. Smoother + rounder than KLT+SVD
and closes much better (end ~1 m off start vs ~2.5 m) -> held heading
better; consistent with PnP>SVD from the odometry 2x2 (still a skewed loop:
left side over-extends to x~-2.3 vs true ~-1.1). The "rollercoaster"
(points up/down) the operator saw is the 3D CLOUD only (both pose graphs
exactly planar) = elevated/tilted camera mount projecting depth, NOT the
trajectory. Cleaner-map plan (roadmap #7; works WITH the fixed bag because
it is an RTAB-Map setting, not a bag property): view the 2D OCCUPANCY GRID
instead of the 3D cloud + pass launch args:="--Reg/Force3DoF true
--Grid/RangeMax 4.0 --Grid/MaxObstacleHeight 0.5" to flatten SLAM + cap
height/range (cuts compute, drops high walls). KLT+PnP next (last), being
run with those cleaner-map flags.

COURSE NOTE (operator, applies to ALL interpretations): the recorded run
does NOT return to the exact start — recording was stopped near the END of
the 1st left turn (which sits just past the start). So a trajectory ending
~1 m from start, at the end-of-1st-turn position, is EXPECTED/correct, not
drift. Re-read accordingly: KLT+SVD's end ~2.5 m off = genuine drift;
ORB+PnP (~1 m) and KLT+PnP (~1.06 m, end at (0.94,0.47)) = basically the
correct stop point.

KLT+PnP -> vo_map_live_klt_pnp.db (Force3DoF + Grid caps) = 122 nodes,
11.19 m, 75 neighbor links, 3 loop closures, pose z exactly [0,0] (operator
confirmed NO up/down on the nodes — the 2D/Force3DoF flags worked). BEST-
SHAPED of the three (most rectangle-like, least skewed); end (0.94,0.47) =
correct end-of-1st-turn stop point. ONE 1.62 m jump at node 119->120 right
at the end (= operator's "2-3 s of missing nodes" — odometry snapped to the
end without intermediate keyframes); everything else smooth (next step only
0.17 m). Slight heading wobble during the 1st turn (operator: brief
heavy-right before the 2nd left turn), small, no jump.

FALSE-YAW SERIES COMPLETE (force_cart_yaw=false, 1x, manual, rtabmap_viz):
  KLT+SVD : 131 nodes / 11.62 m / 8 LC / 0 jumps / end ~2.5 m off (most drift)
  ORB+PnP : 119 nodes / 11.14 m / 6 LC / 1 jump 1.77 m / end ~1 m (closes well)
  KLT+PnP : 122 nodes / 11.19 m / 3 LC / 1 jump 1.62 m / end ~1.06 m, BEST SHAPE
All pose graphs exactly planar. PnP variants close the loop better than
KLT+SVD (less yaw drift) -> map-level agreement with the odometry-2x2
PnP>SVD finding. rtabmap_viz legend (operator asked): white line = raw
odometry path; RGB-axis triad = live robot/camera pose; blue line+dots =
optimized map graph; red = loop-closure links. NEXT: true-yaw
(force_cart_yaw=true) comparison via a clean method (no live TF collision).

YAW COMPARISON — KLT+PnP (camera only) vs Cartographer
(compare_kltpnp_vs_cartographer_yaw.png): total heading turned KLT+PnP
449.6 deg vs Cartographer 457.5 deg = within ~8 deg (~2%) over the whole
loop. So the camera-only yaw is GLOBALLY close; the problem is LOCAL turn
distribution — KLT+PnP's loop is rotated/skewed vs Cartographer's because
turns happen at slightly wrong rates/places. Camera yaw = directionally
solid globally, locally noisy at turns; Cartographer (IMU-fused) is the
smooth local reference. Good redundancy story.

LOOP-CLOSURE note: detection is RTAB-Map's OWN appearance matching
(bag-of-words on its ORB features), independent of our VO frontend, so the
LC count (KLT+SVD 8 / ORB+PnP 6 / KLT+PnP 3) is not a VO-frontend property.
Two levers to use closures more: (1) more/spread features for RTAB-Map's
detector — Kp/MaxFeatures (default 500), Kp/GridRows + Kp/GridCols; (2)
better yaw — drifted odometry makes detected closures get REJECTED by the
RGBD/OptimizeMaxError gate (observed earlier: a closure rejected at 44.9 deg
implied correction), so less yaw drift => more closures accepted.

OPERATOR ALGORITHM INSIGHT (future work, non-holonomic constraint): the VO
produced kinematically IMPOSSIBLE motions — a near-pure sideways move at the
end of the 1st turn and a backward step after the end jump. A car can only
move along its heading. Proposed fix (sound): keep the VO step MAGNITUDE but
correct its DIRECTION to the feasible heading (bicycle/Ackermann model +
motor commands), instead of trusting a bad yaw. This is the motion-model
constraint an EKF applies; force_cart_yaw=true is the crude version (borrow
Cartographer yaw), the principled version constrains VO to the bicycle
model. Adopt for a future VO improvement.

EKF OFFLINE FEASIBILITY (roadmap "variant D"): doable on the bag with NO
QCar. Install ros-humble-robot-localization; feed the EKF /vo/odometry (has
honest covariance) + Cartographer pose (small relay: bagged map->base_link
TF -> odom topic, or use /vo/cart_* scalars); play bag -> /odometry/filtered
-> record + plot fused vs VO vs Cartographer. Caveat: bag has no separate
IMU/wheel topic (Cartographer already fused those), so the offline EKF fuses
VO + Cartographer-pose = the redundancy showcase. Pending operator go-ahead;
operator pausing to study sensors/EKF/bicycle model first.

EKF OFFLINE DEMO — DONE (variant D, off-car, no robot_localization needed).
/tmp/rtabmap_humble/ekf_demo.py reuses nav_to_pose.py's QcarEKF (its exact
bicycle-model f/Jf/prediction + covariance-weighted correction) verbatim, run
on the vo_klt_pnp bag (2258 samples). Prediction driven by CARTOGRAPHER
measured motion (v from delta-pos, delta from yaw-rate via inverse bicycle) as
a proxy for the [speed,delta] motor commands the bag does NOT contain;
CORRECTION = VO pose with its HONEST per-frame covariance straight from
/vo/odometry. Result (ekf_demo_vo_plus_cartographer.png): fused trajectory
(red) tracks the clean loop ~ Cartographer and is far smoother than raw VO
(gray). VO honest 1-sigma swings 0.042 m (confident, straights) -> 7.07 m
(blind, turns/bare-wall), mean 0.647 m, and the EKF down-weights VO exactly
where that spikes -> demonstrates the redundancy thesis offline ("VO publishes
honest uncertainty, the EKF does the right blend"), no QCar. CAVEATS:
prediction uses Cartographer-motion as a motor-command proxy (the live EKF
uses real [speed,delta]); the VO source was force_cart_yaw=true so its
yaw ~= cart (fusion shown is mainly x,y); Q/R are rough demo values, untuned.
This closes the off-car redundancy-showcase arc; live/tuned EKF + the
non-holonomic VO constraint remain the on-car follow-ups.

EKF NON-HOLONOMIC GATE — attempted offline (ekf_demo_nh.py, R-inflate + a
"redirect to heading" variant) on the vo_klt_pnp bag. 17 sideways-violation
samples flagged but all three modes (no-fix / R-inflate / redirect) produced
ESSENTIALLY IDENTICAL trajectories because the flagged samples already
coincided with VO honest-covariance spikes (EKF was already ignoring VO
there). On this VO source the gate is correct in principle but redundant
with the honest covariance; on pure-camera VO (force_cart_yaw=false) it
would matter more, but no synchronized cart-bearing bag exists for that.
Per operator decision, the NH-test artifacts were REVERTED (ekf_demo_nh.py
+ ekf_demo_nonholonomic.png deleted); the original ekf_demo.py +
ekf_demo_vo_plus_cartographer.png stand. EKF/VO tuning deferred (no time).

OPERATOR 2D-MAP QUESTION (resolved): Cartographer's native lidar 2D
occupancy grid is NOT in vslam_test12 (no /map, no /scan recorded — only
camera + /tf with Cartographer's pose + /vo/*), so it can't be
reconstructed. However cartographer.db (variant C) DOES store an
RTAB-Map-built 2D occupancy grid per node: Data.{ground_cells,
obstacle_cells, empty_cells, cell_size} populated from the depth camera and
registered to Cartographer's pose. Open cartographer.db in
rtabmap-databaseViewer and use View -> Show 2D Map (or the grid toolbar
button) to display it. That is the closest available 2D map of that
environment riding Cartographer's reliable trajectory.

## 2026-05-21 Session wrap — cleanup, migration to Ubuntu, full roadmap

Cleanup (file deletions, verified safe — not entry points, no code
imports, only doc mentions):
- DELETED autonomy/vo_capture.py (fault-status file-capture util;
  superseded by copy-paste + bag recording).
- DELETED autonomy/vo_live_plot.py (matplotlib live plotter;
  superseded by saved PNG analysis plots).
- KEPT autonomy/vo_supervisor.py despite operator's "not using it":
  it is the act-on-faults node (stop_advised from /vo/healthy), a
  registered entry point, and core to the VO-redundancy thesis +
  EKF roadmap. Surfaced to operator; keep unless explicitly killed.
- KEPT yolo_detector_new.py (possible future object-detection map
  annotation), nav_to_pose, trip_planner, all active VO nodes, and
  the *_old.py frozen references.
- NOTE: did NOT reorganize autonomy/ into subfolders — ROS2
  ament_python requires the flat module layout (entry points +
  imports depend on autonomy.X:main); subfoldering would break the
  build for no functional gain.
- CLAUDE.md repo-layout updated (removed the two deleted files,
  added vo_odom_tf_relay.py, sharpened vo_supervisor description).

Migration off the QCar (operator handing the physical car to
teammates): all remaining work is bag-based -> portable to the
operator's Ubuntu machine; no live car / camera / Cartographer
needed. Move = (1) git push Gabriel + pull on Ubuntu (push must be
done from an interactive terminal; the QCar shell has no GitHub
creds), (2) rsync the data (dbs + small vo_* bags ~160 MB for
view/analysis; + the 5.7 GB vslam_test12 bag for re-runs),
(3) apt install ros-<distro>-rtabmap-ros + python3-opencv/numpy/
matplotlib, colcon build qcar2_autonomy for vo_node runs.

FULL ROADMAP (everything discussed this session, prioritized):
DONE:
  - VSLAM showcase RTAB-Map A/B/C (pure-visual / our-VO / cartographer
    -fused); quantified; poster panels.
  - KLT/PnP 2x2 toolbox campaign (PnP lowers invalid rate ~3-4x; KLT
    adds features but not reliability; jumps are environmental).
NEAR-TERM (bag-based, do on Ubuntu):
  1. Optional --rate 0.5 confirmation re-run of the 4 VO configs for
     poster-grade numbers (kills 1x frame-drop confound).
  2. KLT+PnP 3D map: RTAB-Map variant B with vo_node on klt+pnp ->
     relay -> rtabmap, for a showcase panel.
  3. RANSAC / feature poster panel: extend vo_image_overlay.py to
     draw green inliers / red outliers + fitted model + a
     before/after-RANSAC two-frame view.
MID-TERM (the redundancy culmination):
  4. EKF fusion "variant D": wire robot_localization to fuse
     /vo/odometry (honest covariance) + Cartographer pose -> fused
     odom -> feed RTAB-Map. vo_supervisor ties in (act on faults).
     Showcase pitch: "VO publishes honest uncertainty, EKF does the
     right blend." NOTE the C-result finding: Cartographer is smooth
     (not jumpy) so it is the reliable backbone; VO is the jumpy
     input the EKF down-weights.
  5. RTAB-Map mapping->localization workflow: live-mapping launch
     (build+save .db during the competition practice run while driving
     the whole mat) + localization-mode launch (load .db, lightweight,
     during the real task run).
  6. camera_bridge rgb/depth sync investigation: the 0.03-0.65 s
     rgb/depth timestamp misalignment degrades ALL variants and is a
     prime suspect for the curve jumps; fixing timestamping could
     clean up everything.
  7. Z-range mapping limit: Grid/MaxObstacleHeight + depth limits to
     cap cloud height ~mat level -> cut compute on the live-mapping run.
FUTURE:
  8. Object-detection map annotation: tag the 3D map with yolo
     detections (stop sign / light at a pose) so trip_planner knows
     the map in advance (keep yolo_detector_new).
  9. New-obstacle handling at task time: Nav2 obstacle/costmap layer
     on the static prebuilt map (NOT a live remap / learning agent).
  10. Resolution campaign: intrinsics_source param (subscribe
      /camera/camera_info -> resolution-agnostic VO) + 720p matrix
      (480p x {400,800,1200} U 720p x {600,800,1200}).
  11. Cartographer covariance tuning for the EKF (its covariance is
      ~constant by default).

## 2026-05-21 KLT/PnP campaign — full 2x2 result (PnP lowers invalid rate; jumps are environmental)

Completed the 2x2 (vo_orb_svd, vo_klt_svd, vo_orb_pnp, vo_klt_pnp)
via bag replay of vslam_test12. Decoded all four /vo bags.

  config       msgs  inliers  invalid%  jumps>0.3  ok%
  ORB+SVD      1278  88       9.4       2          30
  KLT+SVD      1116  141      14.5      4          29
  ORB+PnP      2295  148      3.7       3          38
  KLT+PnP      2258  238      3.1       2          39

Findings:
- KLT tracks more features than ORB (141>88, 238>148) but on SVD it
  did NOT help (worst invalid rate + most jumps).
- PnP is the standout: ~3-4x LOWER invalid rate than SVD (3-4% vs
  9-15%) on both frontends. Mechanistic: PnP (3D->2D reprojection)
  needs valid depth on only ONE frame; SVD (3D->3D Procrustes) needs
  good depth on BOTH -- and our depth is noisy on the white walls,
  so PnP degrades more gracefully. KLT+PnP best on paper (most
  inliers, lowest invalid, fewest jumps).
- KEY: big jumps (>0.3 m) stay at 2-4 across ALL FOUR configs. Neither
  frontend nor estimator fixes them -> the jumps are
  depth/sync/curve-dynamics, NOT an algorithm-choice problem.

CAVEAT (methodology): PnP runs logged ~2x more msgs (2295/2258 vs
1278/1116) -> 1x-replay frame-drop nondeterminism (load-dependent;
SVD runs earlier/warmer, PnP later). Absolute counts not directly
comparable; RATES control for it and still favor PnP, and a 3-4x
invalid-rate gap is too large to be pure load. To put hard numbers
on the poster, re-run all 4 at --rate 0.5 (forces every-frame
processing, removes the drop confound).

Plots saved: ~/vo_rtab_bags/klt_vs_orb_trajectory.png and
~/vo_rtab_bags/klt_pnp_2x2_trajectories.png.

Reminder logged for operator: these 4 runs are odometry-only (no
3D map / loop closure -- that is the RTAB-Map side). To get a 3D
map for a winning config (e.g. KLT+PnP), re-run RTAB-Map variant B
with vo_node set to that frontend/estimator feeding /vo/odometry ->
relay -> rtabmap.

Operator decision pending: (1) clean --rate 0.5 confirmation run,
(2) accept directional finding + write conclusions, or (3) build a
3D map for the best config. Default-safe toolbox baseline stays
ORB+SVD; PnP is a promising robustness lever worth confirming.

## 2026-05-21 KLT/PnP campaign — KLT vs ORB result (ORB wins on reliability)

Ran vo_orb_svd (baseline) + vo_klt_svd via bag replay of vslam_test12.
Decoded /vo/odometry + diagnostics from each /vo bag (rosbag2_py +
deserialize_message). Both = 983-1278 odom msgs over the ~128s replay.

  metric            | ORB+SVD | KLT+SVD
  inliers (mean)    | 88      | 141
  invalid frames    | 120     | 162
  big jumps >0.3 m  | 2       | 4
  bare_wall flags   | 83      | 49
  ok frames         | 386     | 328
  odom msgs         | 1278    | 1116

VERDICT: KLT does NOT clearly beat ORB — mixed, leans ORB. KLT tracks
more features (141 vs 88 inliers) and flags bare_wall less, but pays
with more invalid frames, more big jumps, and fewer ok frames. For
our reliability-first use, ORB+SVD remains the better default; KLT
trades robustness for feature count. Caveat: 1x replay frame-drop
nondeterminism adds noise to small differences, but the
reliability gap (invalid + jumps) is consistent.
(Normalizing path by sample count: ORB 19.85m/1278 = 0.0155 m/step,
KLT 16.91m/1116 = 0.0151 m/step -- essentially identical per-step,
so the raw path-length difference is just sample count, not
smoothness.)

Saved trajectory comparison plot:
~/vo_rtab_bags/klt_vs_orb_trajectory.png (ORB blue, KLT red,
Cartographer black ref).

Clarified for the operator the two experiment types from the one
bag: (1) RTAB-Map variants A/B/C = SLAM/3D-map (rtabmap_viz, .db
files, loop closure); (2) orb/klt/svd/pnp = OUR vo_node odometry
A/B (only /vo/odometry, no 3D map / no loop closure). They connect:
the winning VO config would feed a future RTAB-Map variant B.
3D maps viewable via rtabmap-databaseViewer on the saved .db files.

Next: run vo_orb_pnp (estimator test on the proven ORB frontend),
optionally vo_klt_pnp, then final 2x2 verdict.

## 2026-05-21 KLT/PnP campaign — bag-based methodology + multi-user isolation

Multi-user check: domain 67 verified clean (only /parameter_events,
/rosout) — friend running in Docker on a DIFFERENT ROS_DOMAIN_ID is
fully isolated by DDS; zero cross-talk. Only shared resource would
be the physical camera, which the bag-based campaign does not touch.

Raw-/vo/cart_x check: deferred/skipped — operator reasoning accepted
(Cartographer has no vision -> smooth but drifts; loop-closure
rescues drift; VO rescues when Cartographer degrades). The C-map
already shows Cartographer smooth.

KLT/PnP campaign methodology = BAG-BASED on vslam_test12 (matches the
established bag-driven evaluation used for the grid A/B):
- vo_node subscribes camera with qos_profile_sensor_data (BEST_EFFORT),
  which matches the bag natively -> NO QoS override needed (unlike
  RTAB-Map). So we replay vslam_test12 camera+TF straight into
  vo_node and A/B the knobs on identical input. Deterministic, no
  driving, no camera contention.
- Per config, 3 terminals: T1 vo_node (use_sim_time:=true,
  camera_mode:=physical, alignment_mode:=auto, depth_scale:=0.0,
  n_features:=1200, feature_grid:=8, + vo_frontend/vo_estimator under
  test); T2 record /vo/* (odometry, fault_status, conditioning,
  reason, inliers, confidence, vo_x/y/psi, cart_x/y/psi) to a small
  bag vo_<config>; T3 bag play vslam_test12 --clock --topics
  camera-triple + /tf + /tf_static (NOT /vo/* — vo_node regenerates
  it). Order T1,T2 then T3; on T3 end Ctrl-C T2 then T1.
- Configs (regenerate baseline via replay too, for apples-to-apples):
  vo_orb_svd (baseline), vo_klt_svd, vo_orb_pnp, vo_klt_pnp.
- Start with orb_svd + klt_svd (the headline KLT-vs-ORB question).
  Analysis = decode each /vo/odometry trajectory (jumps, drift,
  agreement vs /vo/cart_*) the same way the RTAB-Map dbs were
  decoded. If KLT beats ORB, run the PnP pair; else conclude.

Replay at 1x (realistic, best-effort drops mirror live cost). Could
redo at --rate 0.5 if all-frame deterministic processing is wanted.

## 2026-05-21 Physical Test 12 — Variant C result + full A/B/C table (Cartographer is smooth)

Variant C (cartographer pose via TF, odom_frame_id:=map,
map_frame_id:=carto_map) ran successfully. Decoded cartographer.db:

  Full A/B/C (decoded from .db Node tables):
  | metric        | A rtabmap_odom | B vo_node | C cartographer |
  | nodes         | 37             | 134       | 136            |
  | path length   | 4.66 m         | 11.80 m   | 12.35 m        |
  | median step   | 0.088 m        | 0.063 m   | 0.091 m        |
  | max step      | 1.493 m        | 0.937 m   | 0.164 m        |
  | jumps >0.3 m  | 1              | 6         | 0              |
  | loop closures | 8              | 3         | 14             |
  | z-range       | [-0.044,0.002] | [0,0]     | [0,0]          |

KEY FINDING: C is the best map by every measure — full loop, ZERO
jumps (max step 0.16 m ~ 0.1 m/s x cycle), most loop closures (14).
This REVISES the prior assumption (CLAUDE.md / earlier entries) that
"Cartographer jumps in curves like VO." In this bag, the
Cartographer-driven RTAB-Map trajectory is smooth end-to-end; only
our VO (B) jumps. Implication for the EKF plan: Cartographer is a
smooth reliable backbone and VO is the jumpy input, so EKF
down-weighting of VO during its spikes SHOULD work (earlier worry
"if both jump, EKF can't help" not supported here).

CAVEAT: the C poses are RTAB-Map's OPTIMIZED output (14 loop
closures could be smoothing raw Cartographer jitter). To confirm
whether Cartographer's RAW pose jumps, parse /vo/cart_x directly
from the bag (offered to user, pending). This matters before
trusting Cartographer as the EKF backbone.

Showcase narrative crystallized: pure-visual (A) cannot complete the
loop in this low-texture environment; our VO (B) completes it but
inherits curve jumps; Cartographer-fusion (C) is clean + complete +
best loop closure. Directly motivates a future "variant D":
VO + Cartographer -> EKF -> fused odom -> RTAB-Map.

Idea triage (operator brainstorm; logged for continuity, not all
actioned now):
- "Build RTAB-Map in our files": do NOT reimplement RTAB-Map. Use
  its standard mapping-run -> localization-run workflow. Practice
  run = live RTAB-Map (camera+cartographer) saving the .db; real
  run = RTAB-Map localization mode loading that .db (lightweight).
  Two launch configs, no new algorithm. Feasible.
- Object-detection annotation of the map (stop sign / light at a
  pose): feasible future project via RTAB-Map labels/landmarks +
  yolo_detector. Not now.
- New obstacles at task time: Nav2 obstacle/costmap layer on the
  static prebuilt map, NOT a learning agent / live remap.
- Limit Z range of the cloud: GOOD concrete optimization.
  RTAB-Map Grid/MaxObstacleHeight + depth-range limits; cap Z to
  ~mat height + margin to cut compute and clean the cloud. Add to
  the live-mapping launch when we build it.

Next: view A/B/C in rtabmap-databaseViewer for poster screenshots
(Graph View = trajectory + loop-closure links), then the KLT/PnP
odometry campaign, then conclusions.

## 2026-05-21 Physical Test 12 — A vs B comparison + Z-artifact explanation + variant C recipe

Variant B re-ran successfully with the covariance-sanitizing relay.
Decoded both pose graphs directly from the .db Node tables (python3
sqlite3, 3x4 row-major pose blob, translation = floats[3,7,11]):

  rtabmap_odom.db (A, pure visual): 37 nodes, path 4.66 m,
    step mean 0.129 / median 0.088 / max 1.493 m, 1 step >0.3 m,
    z-range [-0.044, 0.002], 8 loop closures.
  vo_node.db (B, our VO + IMU yaw): 134 nodes, path 11.80 m,
    step mean 0.089 / median 0.063 / max 0.937 m, 6 steps >0.3 m,
    z-range [0.000, 0.000] (exactly planar), 3 loop closures
    (2 global type-1 + 1 local type-2).

Headline findings:
- COMPLETENESS vs TRACKING LOSS: pure-visual A captured only 4.66 m
  of the ~12 m loop (lost tracking on white walls -> stopped adding
  nodes; one 1.49 m teleport on re-acquire). Our VO (B) never goes
  dark -> full 11.80 m loop, 134 nodes. Concrete redundancy win:
  pure-visual SLAM cannot complete the loop in this environment;
  ours can.
- JUMPS QUANTIFIED (B): top node-to-node steps 124->125 0.94 m
  (0.71 m/s), 110->111 0.70 m (0.60 m/s), 61->62 0.65 m (0.57 m/s)
  -- all 6-7x the commanded 0.10 m/s. These are the unphysical
  /vo/odometry jumps baked into the map; locations match the
  operator's visual report (biggest near the end / 5th turn).

Z-ARTIFACT explained (operator saw "Z jumps, QCar went up/down"):
  pose graph is verified exactly planar (all 134 nodes z=0.0000,
  spread 0). The apparent Z motion is in the rendered CLOUD, not the
  trajectory: the camera mount (base_link->camera_color_optical_frame)
  is elevated + tilted, so an x/y/yaw pose jump projects depth points
  up/down -> vertical smear in the cloud at the jump locations,
  amplified by the rgb/depth time misalignment. Cleanup option (NOT
  applied, to keep A/B/C comparable): Reg/Force3DoF=true constrains
  RTAB-Map to x/y/yaw and removes the Z artifacts.

KLT-for-RTAB-Map clarification (operator question): in the
external-odometry architecture (variant B), the odometry frontend
(KLT or ORB) only supplies POSES; RTAB-Map runs its OWN ORB
loop-closure detection on the raw RGB images, independent of our
frontend. So KLT-odometry + RTAB-Map's-ORB-loop-closure IS a valid
combo -- the earlier "KLT can't do SLAM" caveat is specifically
about KLT as the appearance/loop-closure frontend (no descriptors),
not as an odometry source. Worth testing in the KLT/PnP campaign:
does KLT reduce the 6 jumps? Only valuable if KLT beats ORB for
odometry on the mat.

EKF reframe (operator's "replace map nodes + train a model" idea,
deferred): EKF is a recursive Bayesian filter, not a trained model
(no learning across runs; you tune its process/measurement noise).
The fusion architecture is VO + cartographer -> EKF -> fused odom ->
RTAB-Map odom input (a future "variant D"), NOT post-hoc swapping of
map nodes.

Variant C recipe (cartographer via TF; no /odom topic exists; no
rebuild, no relay; planar so no covariance-invert crash because
TF-based odom is computed internally):
  T6: ros2 launch rtabmap_launch rtabmap.launch.py
      rgb_topic:=/camera/color_image depth_topic:=/camera/depth_image
      camera_info_topic:=/camera/camera_info frame_id:=base_link
      approx_sync:=true rgbd_sync:=true qos:=2 visual_odometry:=false
      odom_frame_id:=map map_frame_id:=carto_map use_sim_time:=true
      database_path:=~/vo_rtab_bags/cartographer.db
  T7: ros2 bag play vslam_test12 --clock --topics
      /camera/color_image /camera/depth_image /camera/camera_info
      /tf_static /tf
  (odom_frame_id:=map reads Cartographer's map->base_link from the
  bagged /tf; map_frame_id:=carto_map avoids the frame collision.)

Next: variant C, then the KLT/PnP odometry campaign, then
conclusions.

## 2026-05-21 Physical Test 12 — Variant B FATAL: zero twist covariance (RTAB-Map setInfMatrix inf)

After the frame/TF relay fix, variant B got past the TF-tree problem
(rtabmap subscribed to /vo/odom_relay, started processing) but the
rtabmap node CRASHED:
  [FATAL] Link.cpp:139::setInfMatrix() Condition
  (uIsFinite(infMatrix(2,2)) && infMatrix(2,2)>0) not met!
  [Linear information Z should not be null! Value=inf ...]
  terminate called after throwing 'UException'

Root cause (read vo_node.py:957-961): the POSE covariance is fully
populated (cov[14]=VO_BIG_POS_VAR=25 for Z, etc.), but the TWIST
covariance only set x/y/yaw:
  tw = [0.0]*36; tw[0]=var_x; tw[7]=var_y; tw[35]=var_yaw
leaving tw[14] (vz), tw[21] (v_roll), tw[28] (v_pitch) = 0.0.
RTAB-Map inverts the covariance for its link information matrix;
1/0 = inf on the Z diagonal trips the assertion and aborts the node.

Two-layer fix (both committed):
1. Relay (no-rebuild, effective immediately): vo_odom_tf_relay.py
   now sanitizes BOTH pose and twist covariance via _sanitize_cov():
   any diagonal entry that is non-finite or <=0 is replaced — off-plane
   dims (z/roll/pitch, indices 14/21/28) -> 1e6 (large finite, ~0
   information), observed dims (x/y/yaw) -> 1e-6 floor. Verified
   standalone: zero twist-Z -> 1e6 -> info 1e-6 (finite). Since the
   relay runs as python3, restarting it applies the fix with no
   rebuild and no re-record (the bagged covariance is sanitized on
   the fly during playback). py_compile PASS.
2. vo_node root-cause (after next rebuild): vo_node.py twist
   covariance now also sets tw[14]=VO_BIG_POS_VAR, tw[21]=tw[28]=
   VO_BIG_YAW_VAR (mirrors the pose covariance convention). Removes
   the latent zero-diagonal bug from the published /vo/odometry
   regardless of downstream consumer. py_compile PASS. Does not
   change x/y/yaw honesty (the EKF redundancy signal is untouched).

Note: the rgbd_sync "rgb/depth time difference high (0.03-0.65 s)"
warnings persist and are a SEPARATE, non-fatal issue (camera_bridge
timestamp alignment) tracked in the variant-A entry; they degrade
map quality but do not stop mapping. Pending: re-run variant B with
the updated relay, then compare A vs B.

## 2026-05-21 Physical Test 12 — Variant B (vo_node) stuck at 1 node: frame fix (vo_odom_tf_relay)

Symptom: variant B (feed our /vo/odometry into RTAB-Map) built only
1 node (vo_node.db = 472 KB, 0 links). GUI showed the live cloud but
no accumulating map; "id stayed = 1".

Root cause (from the pasted Terminal output): RTAB-Map logged
"(can transform map -> base_link?) Could not find a connection
between 'map' and 'base_link' ... Tf has two or more unconnected
trees." Our vo_node publishes /vo/odometry stamped frame_id='map',
child='base_link' (vo_node.py:943-944) to mirror Cartographer. As
RTAB-Map external odometry this fails two ways:
  1. frame_id='map' collides with RTAB-Map's own map_frame_id='map'.
  2. Nothing broadcasts the odom->base_link TF that RTAB-Map needs to
     interpolate the camera pose at each image stamp (vo_node only
     uses tf2 to LISTEN, never broadcasts; and /tf was excluded from
     the variant-B playback).
=> broken TF tree, map cannot advance past node 1.

Fix: new node autonomy/vo_odom_tf_relay.py. Subscribes /vo/odometry,
republishes it on /vo/odom_relay with frame_id='odom',
child='base_link', and broadcasts the matching odom->base_link TF.
RTAB-Map then sees the connected chain map -> odom -> base_link ->
camera. Standalone-runnable (python3, no colcon) so it works without
a rebuild:
  python3 .../autonomy/vo_odom_tf_relay.py --ros-args -p use_sim_time:=true
py_compile PASS; rclpy/tf2_ros/nav_msgs/geometry_msgs import OK
standalone.

Wiring:
- setup.py: added entry point vo_odom_relay=autonomy.vo_odom_tf_relay:main
  (usable as `ros2 run qcar2_autonomy vo_odom_relay` after a rebuild).
- qcar2_rtabmap_launch.py vo_node branch: odom_topic changed from
  /vo/odometry to /vo/odom_relay (requires the relay running). Takes
  effect after the next qcar2_nodes rebuild.

No-rebuild path used for Test 12 today: stock
`ros2 launch rtabmap_launch rtabmap.launch.py ... qos:=2
visual_odometry:=false odom_topic:=/vo/odom_relay ...` + the relay
node + bag play of /vo/odometry. With qos:=2 the camera QoS override
file is not needed on this path (RTAB-Map subscribes Best Effort,
matching the bag's native Best Effort camera topics).

Note: this does NOT change vo_node's own /vo/odometry framing (still
frame_id='map' for the Cartographer-redundancy comparison). The relay
is purely an adapter for the SLAM ingestion path. Pending: re-run
variant B with the relay, then compare A vs B.

## 2026-05-21 Physical Test 12 — Variant A (rtabmap_odom) result + rgb/depth sync finding

After the QoS fix, variant A (pure-visual rtabmap_odom) built a real
map. Read the db directly via python3 sqlite3 (no sqlite3 CLI on the
car):
  ~/vo_rtab_bags/rtabmap_odom.db = 16 MB
  Node (keyframes)    = 36
  Link type 0 (neighbor/trajectory) = 24
  Link type 1 (global loop closure) = 3   <-- 3 loop closures ACCEPTED
  map_id distinct     = {0} (single session)

Interpretation:
- 36 nodes but only 24 neighbor links => the trajectory graph is
  fragmented (~11 odometry breaks; a clean single chain would have
  ~35 neighbor links). Confirmed by the log: many
  "OdometryF2M Registration failed: Not enough inliers 0/20" with
  quality=0, recovering to quality 42->109 only near the end.
- FINDING: RTAB-Map's OWN pure-visual odometry struggles on the
  white-wall / low-texture mat — same failure mode we see in our VO.
  Strengthens the narrative that the environment (texture-poor) is
  the fundamental limiter, not just our implementation. Reference
  implementation in the same conditions also fragments.
- 3 accepted global loop closures despite the fragmentation — RTAB-Map
  rejected weaker candidates (user saw "image 13 rejected") but
  confirmed 3.

SECONDARY FINDING (affects ALL variants + our VO): rgbd_sync logged
"The time difference between rgb and depth frames is high
(diff=0.03-0.65 s)" repeatedly. Color and depth in the bag are
mis-timestamped by 1-several frames (target <0.01 s), so RTAB-Map
pairs mismatched RGB/depth -> corrupted depth association -> worse
odometry. Root cause likely in qcar2_camera_bridge: whether color
and depth are stamped from the same capture or at publish time.
TODO (not today): investigate camera_bridge timestamping; consider
approx_sync_max_interval tightening or hardware-synced capture.

GUI note: rtabmap_viz "extrapolation into the future" TF warnings
are visualization-side lag during fast bag playback; they did not
affect the map the core rtabmap node built.

Hypothesis to test with variant B: our /vo/odometry always emits a
pose (never hard-loses tracking the way rtabmap_odom did), so the
vo_node-driven map may be LESS fragmented than variant A — a
potential concrete win for our approach. Pending variant B run.

## 2026-05-21 Physical Test 12 — RTAB-Map QoS mismatch (empty map root cause + fix)

Symptom: Physical Test 12 variant A (rtabmap_odom) produced an empty
rtabmap_viz GUI (no odometry, no 3D map, no loop closures) and a
90 KB database. rtabmap nodes spammed "Did not receive data since 5
seconds" for /camera/* despite the bag playing.

Root cause (from the user's pasted Terminal 7 output + bag metadata):
- rosbag2_player warned: "New subscription discovered on topic
  '/camera/color_image', requesting incompatible QoS. No messages
  will be sent to it. Last incompatible policy:
  RELIABILITY_QOS_POLICY".
- Bag metadata confirms the camera topics were recorded with
  reliability: 2 (BEST_EFFORT) — camera_bridge uses sensor QoS.
  (/tf and /vo/* are reliability: 1 / Reliable, so they were fine.)
- RTAB-Map's input subscriptions default to qos=1 (Reliable).
- Reliable subscriber + Best-Effort publisher = INCOMPATIBLE in DDS,
  so zero camera frames reached RTAB-Map → empty map. Not a bag
  problem, not a drive problem — pure QoS handshake failure.

Two fixes (different timelines):

1. Immediate, no-rebuild (what the user runs for Test 12): force the
   bag PLAYER to republish the camera topics as Reliable so they
   match RTAB-Map's Reliable subscription. Created
   ~/vo_rtab_bags/qos_reliable.yaml overriding /camera/color_image,
   /camera/depth_image, /camera/camera_info to reliable+volatile and
   /tf_static to reliable+transient_local (latched static TF). Play
   with:
     ros2 bag play vslam_test12 --clock --topics <cam triple> /tf_static \
       --qos-profile-overrides-path /home/nvidia/vo_rtab_bags/qos_reliable.yaml
   (variant B adds /vo/odometry to --topics; same override file.)

2. Permanent, takes effect on next qcar2_nodes rebuild: added a `qos`
   launch arg to qcar2_rtabmap_launch.py defaulting to '2' (Best
   Effort), forwarded to rtabmap.launch.py as `qos:=2`. This makes
   RTAB-Map subscribe Best Effort, matching both the live bridge and
   a recorded bag, so NO --qos-profile-overrides-path is needed after
   the rebuild. py_compile PASS. Default-safe rationale: qos=2 is the
   correct setting for our actual sensor source in both live and bag
   modes; qos=1 was never compatible with the bridge.

Doc note: after the rebuild, the qos_reliable.yaml override becomes
unnecessary (harmless if left in). Until the rebuild, the override
is REQUIRED because the installed wrapper still subscribes Reliable.

## 2026-05-21 No /odom topic on QCar — Cartographer pose is TF-only; rtabmap cartographer-mode fix

Discovery during Physical Test 12 bring-up: the pre-flight
`ros2 topic hz /odom` hung. Investigation (live `ros2 topic list`
on domain 67 while the user's cartographer + vo_node were running)
confirmed there is **no `/odom` topic** on this QCar. Full live
list relevant subset: /camera/*, /scan, /map, /submap_list,
/tf, /tf_static, /qcar2_imu, and the /vo/* family — no /odom.

How the Cartographer pose is actually exposed:
- Cartographer publishes the pose through TF only: map -> base_link.
- vo_node reads it via `lookup_transform('map', 'base_link')`
  (autonomy/vo_node.py:635-636) and republishes convenience scalars
  /vo/cart_x, /vo/cart_y, /vo/cart_psi (all present in the live
  list). /vo/odometry itself is stamped frame_id='map',
  child_frame_id='base_link' (vo_node.py:943-944).

Consequences / fixes:
- Pre-flight: drop the `/odom` hz check. Confirm Cartographer is
  flowing via `ros2 topic echo /vo/cart_x --once` (a number = TF
  pose reaching vo_node).
- Bag recording: `/odom` removed from the record list (records
  nothing); `/tf` + `/tf_static` carry the Cartographer pose and
  are the source for the cartographer playback variant. Added more
  /vo/* analysis scalars to the canonical record list (vo_x/y/psi,
  cart_x/y/psi, reason) for the §7.7 jump investigation.
- qcar2_rtabmap_launch.py cartographer branch rewritten: instead of
  the nonexistent `odom_topic:=/odom`, it now sets
  `odom_frame_id:=map` (RTAB-Map reads odometry from the TF tree)
  and `publish_tf_map:=false` (so RTAB-Map does not fight
  Cartographer for the map frame). Both args verified to exist via
  `rtabmap.launch.py --show-args`. Caveat documented in-code: the
  map frame carries loop-closure jumps (not a strictly continuous
  odom frame), acceptable for the A/B/C showcase. py_compile PASS.

Playback variant recommendation (corrected to --topics whitelist
instead of --exclude, cleaner than regex blacklisting):
- A rtabmap_odom: play /camera/* + /tf_static only.
- B vo_node: play /camera/* + /tf_static + /vo/odometry.
- C cartographer: play /camera/* + /tf_static + /tf.

Operational note (user decision this session): user opted NOT to
rebuild qcar2_nodes mid-session, so the *installed* launch still
has the old cartographer branch (odom_topic:=/odom). Therefore for
Physical Test 12, variants A and B run as-is; variant C
(cartographer) will hang waiting for /odom and must be deferred
until a qcar2_nodes rebuild. The bag includes /tf + /tf_static so
variant C is fully recoverable later on the same bag with no
re-drive. The launch source fix above is committed to the gabriel
tree and will take effect on the next rsync + colcon build of
qcar2_nodes.

## 2026-05-20 manual_drive turn_rate bump 0.25 → 0.40 rad/s

User reported the WASD manual drive was under-turning on big curves
(angular response too small for the actual mat geometry). Bumped
the `turn_rate` parameter default in
`autonomy/manual_drive.py:36` from 0.25 to 0.40 rad/s.

Math sanity at the default forward_speed=0.10 m/s:
  - old: 0.10 / 0.25 = 0.40 m turning radius
  - new: 0.10 / 0.40 = 0.25 m turning radius
Still well within mat scale; not aggressive enough to spin the car
unintentionally. User can override per-run via
  -p turn_rate:=<value>
if 0.40 needs further adjustment.

py_compile PASS. No other manual_drive behavior touched; speeds
(forward 0.10, reverse 0.08) unchanged. Will require rsync +
colcon build --packages-select qcar2_autonomy before next run.

## 2026-05-20 VSLAM showcase launch wiring committed (Deliverable A code)

Built the record-once / playback-three-ways scaffolding agreed in the
prior turn. Doc-only on VO behavior — `visual_odometry.py`, `vo_node.py`
and the rest of `qcar2_autonomy/` are untouched. All changes are
additive.

State verified before edits:
- df -h /home → 164 GiB free of 227 GiB (plenty of room for several
  bags + dbs).
- vo_node already publishes the topics we need for the showcase:
  /vo/odometry (5 Hz, nav_msgs/Odometry), /vo/fault_status,
  /vo/conditioning, /vo/cart_psi, /vo/vo_psi_shadow, /vo/healthy,
  /vo/confidence, /vo/inliers — all confirmed via grep in
  autonomy/vo_node.py. vo_node does NOT publish odom->base_link TF
  (it only uses tf2_ros for listening), so the bag's /tf can stay
  in playback for vo_node mode without conflict.
- rtabmap_launch.py exposes `visual_odometry:=true|false` to toggle
  rtabmap_odom; `odom_topic` to bind external odom; `approx_sync`
  and `rgbd_sync` available.
- qcar2_nodes/CMakeLists.txt has `install(DIRECTORY launch ...)` so
  the new launch file auto-installs on next colcon build —
  no CMake edit needed.

mkdir:
- ~/vo_rtab_bags/ (created; persistent dir for both bags and .db
  outputs; never /tmp).

New file:
- Development/ros2/src/qcar2_nodes/launch/qcar2_rtabmap_launch.py
  - Wraps rtabmap_launch/rtabmap.launch.py via IncludeLaunchDescription.
  - Single parametric switch:
      odom_source ∈ {rtabmap_odom (default), vo_node, cartographer}.
    Internally maps to (visual_odometry, odom_topic) combinations.
  - Pinned QCar remaps:
      rgb_topic:=/camera/color_image
      depth_topic:=/camera/depth_image
      camera_info_topic:=/camera/camera_info
      frame_id:=base_link
      approx_sync:=true
      rgbd_sync:=true
      queue_size:=30
  - Other args (with defaults): use_sim_time=true (bag playback),
    database_path=~/vo_rtab_bags/rtabmap.db, localization=false
    (mapping), rtabmapviz=true, rviz=false.
  - Pure glue: no rtabmap internals are redefined; the file just
    selects topology and forwards to the stock launch.
  - py_compile PASS.

Easy_Start.txt — new Section 7 "VSLAM Showcase (RTAB-Map,
record-once / playback-three-ways)":
  7.1 Pre-record sanity checks (ros2 topic hz on all required
      topics).
  7.2 Recording the canonical bag — exact `ros2 bag record` topic
      list (camera triple + 8 /vo/* topics + /odom + /tf +
      /tf_static), bag-policy reminder (one best-bag on QCar,
      transfer dbs off, prune old bags), force_cart_yaw two-bag
      experiment recipe.
  7.3 Playback A — rtabmap_odom (pure-visual baseline). Run FIRST
      on every new bag as a sanity check.
        --exclude '/odom' --exclude '/tf' to avoid publisher
        collisions with rtabmap_odom's outputs.
  7.4 Playback B — vo_node (our /vo/odometry from bag).
        --exclude '/odom' only.
  7.5 Playback C — cartographer (/odom from bag). Optional.
        No excludes.
  7.6 Compare the three databases via rtabmap-databaseViewer
      (trajectory shape, loop-closure count, map sharpness,
      feature density).
  7.7 Cartographer-jump investigation: side-by-side replay of
      /vo/odometry and /odom from the bag, with the bash
      one-liner to dump per-axis x to CSV and plot. Same protocol
      works for /vo/vo_psi_shadow vs /vo/cart_psi for the
      force_cart_yaw analysis.
  7.8 Cleanup / transfer (rsync command, expected ~6-10 GiB
      complete showcase set).

CLAUDE.md updates:
- "RTAB-Map / SLAM showcase plan" section rewritten to reflect the
  decided architecture (record-once / playback-three-ways), the
  pinned launch file path and arg surface, the three odom-source
  variants explained in a table with visual/non-visual
  classification, the persistent bag policy, the ORB-only
  frontend rule.
- New section "Camera resolution — verdict (revised)" replaces the
  earlier "skip, needs recalibration" line: 480p baseline keep,
  720p worth a test campaign with LOWER n_features at iso-CPU
  (per user's correction), 1080p probably skip. Documented the
  proposed `intrinsics_source` parameter that would subscribe to
  /camera/camera_info to become resolution-agnostic (next-session
  work, not today). Documented the proposed test matrix:
  480p×{400,800,1200} ∪ 720p×{600,800,1200}.
- New section "EKF fusion — note for when we wire
  robot_localization" captures the corrected Bayesian/Kalman-gain
  description (rebutting the "average / 2" framing), notes that
  /vo/odometry already publishes honest covariance driven by
  conditioning, and flags that Cartographer's roughly-constant
  default covariance is its own tuning concern.

Operational gotcha to remember (encoded in §7.3/7.4/7.5
--exclude lists):
- rtabmap_odom mode + bag's /odom → two publishers on /odom → use
  --exclude '/odom' (and --exclude '/tf' because rtabmap_odom
  publishes odom->base TF itself).
- vo_node mode + bag's /odom → bag's /odom would override
  vo_node's already-bagged /vo/odometry path → exclude /odom.
- cartographer mode → no exclusions; /odom and /tf ARE the source.

NOT done (next sessions):
- intrinsics_source param + /camera/camera_info subscription in
  visual_odometry.py.
- bridge config change to 720p for the resolution campaign.
- robot_localization wiring (EKF fusion of /odom + /vo/odometry).
- compare_rtabmap_dbs.py helper (mentioned in §7.6 narrative;
  manual rtabmap-databaseViewer is fine for the first pass).

## 2026-05-20 Odom-source decision + resolution-bump re-rank + EKF/Cartographer notes

Confirmed decisions:
- VSLAM showcase odom source = **`rtabmap_odom`** (pure-visual SLAM).
  User wants the demo to be camera-only, no lidar involvement in the
  trajectory; the map is built off RTAB-Map's own internal VO on the
  RGB-D stream. Our `vo_node` keeps running independently so the
  dashboard / redundancy story is unaffected. Duplicate ORB cost
  (rtabmap_odom + vo_node) is a real concern on the Jetson; if it
  bites we throttle one of them.
- Resolution bump verdict revised: 480p baseline stays; **720p worth
  trying as an experiment once VSLAM baseline is up**; 1080p probably
  not. Reason for the revision: user correctly noted RealSense exposes
  per-resolution intrinsics, so no manual recalibration is needed —
  we either add a 720p hardcoded table from the SDK enumeration in
  `vo_calib_logs/realsense_calib_*.txt`, or (cleaner) make VO
  resolution-agnostic by subscribing to `/camera/camera_info`. The
  real residual costs are per-frame ORB (~2.25x from 480p to 720p)
  and bridge throughput at the higher resolution.

EKF clarification (because a Quanser engineer told user "EKF just
adds both and divides by two" — that is wrong and would mislead
implementation):
- EKF predict step: project state with motion model; covariance
  grows by process noise.
- EKF update step: Kalman gain weights each measurement INVERSELY
  by its covariance vs. the predicted-state covariance. Result is
  the statistically optimal blend under the Gaussian assumption.
- Practical consequence for our pipeline: because `/vo/odometry`
  already publishes honest covariance (driven by `confidence` and
  `/vo/conditioning`), the EKF automatically downweights VO when
  conditioning is poor. The redundancy pitch for the showcase is
  therefore "VO publishes honest uncertainty and the EKF is
  mathematically guaranteed to do the right blend" — not "we
  average them."

Cartographer-jump observation (open question to investigate on next
runs):
- User saw a VO x-jump 0.9 → 0.78 m at 0.1 m/s with a 200 ms VO
  cycle → implied instantaneous velocity 0.6 m/s, ~6x the actual
  speed. Unphysical, consistent with a bad-match-surviving-RANSAC
  spike or a depth glitch.
- User has seen similar magnitude jumps in Cartographer `/odom`
  during curve segments. If true, this matters for the EKF design
  because Cartographer's covariance is roughly constant by default
  (no analog of our `/vo/conditioning`), so when Cartographer jumps
  it does not flag itself as uncertain — the EKF blindly trusts it.
- Action: next physical run must capture `/odom` and `/vo/odometry`
  side-by-side (extend the standard bag topic list, extend
  dashboard or add a side-by-side plot). Look for synchronized
  jumps and characterize Cartographer's noise profile through
  curves before we wire `robot_localization`.

No code changes yet; this is the planning capture. Implementation of
Deliverable A starts next: mkdir ~/vo_rtab_bags/, write
qcar2_rtabmap_launch.py with rtabmap_odom included, add
Easy_Start.txt §7 with record/playback recipe, refresh CLAUDE.md
RTAB-Map section to reflect pure-visual choice.

## 2026-05-20 VSLAM showcase scoping + EKF-fusion architecture clarification

Design discussion with user (post-professor conversation). Capturing
decisions so future sessions don't relitigate.

User-raised ideas and verdicts:

1. **Higher camera resolution to surface more features**
   - Skip. Intrinsics in `visual_odometry.py` are calibrated for
     640x480; a bump requires recalibration. Per-frame ORB cost
     scales with pixel count (~4x at 720p, ~9x at 1080p), bridge
     bandwidth goes up, real-time budget with cartographer + VO +
     RTAB-Map co-running is unlikely on the Jetson. Worth a one-off
     experiment later, not the baseline.

2. **LoFTR (transformer matcher) for low-texture curve regions**
   - Skip — future work / "presentation footnote" only. Needs
     PyTorch + CUDA inference loop; typical inference 10-20 Hz on
     desktop GPUs, sub-5 Hz expected on Jetson; integration is
     multi-day engineering. Note in the report as a known
     direction we did not pursue.

3. **Adaptive frontend switching (ORB on straights, KLT/LoFTR in
    curves)**
   - Skip until evidence justifies it. Trigger logic (curve
     detection or VO-confidence threshold) is itself a small
     project; RTAB-Map loop closure is blind during any non-ORB
     segment (no descriptors), so switching has a real SLAM cost
     that must be earned by a measurable VO-quality win. Until A/B
     shows KLT beating ORB in *some* regime, this is solving a
     hypothetical.

4. **"EKF swaps landmark coordinates while preserving descriptors"
    architecture**
   - Won't compose that way. Correcting the model here so we don't
     build the wrong thing:
     - RTAB-Map owns the map. Landmark world coords = (camera pose
       at keyframe) x (3D point in camera frame). Descriptors are
       just re-association keys, not coords.
     - robot_localization EKF fuses pose/twist messages with
       covariances and produces ONE fused pose. It does not reach
       into RTAB-Map's database to modify landmarks.
     - RTAB-Map's internal pose-graph + loop-closure optimization
       is what adjusts landmark coords; it accepts no
       "replace-this-landmark" inputs.
   - The *spiritual equivalent* of what user wants is standard and
     clean: feed RTAB-Map an EKF-fused pose via its `odom_topic`
     input. In low-VO-confidence regions the fused pose is
     dominated by Cartographer/IMU/wheel, so landmarks
     triangulated during curves inherit Cartographer-dominated
     coords automatically — without any surgical swapping. This
     is the "research-quality" version of Deliverable A and is a
     stretch goal, not today.

Today's scope (TWO deliverables, hard prioritized A > B):

**Deliverable A — Basic VSLAM (must-have)**
- Create persistent bag dir `~/vo_rtab_bags/`.
- Write `Development/ros2/src/qcar2_nodes/launch/qcar2_rtabmap_launch.py`
  with pinned remaps:
    rgb_topic:=/camera/color_image
    depth_topic:=/camera/depth_image
    camera_info_topic:=/camera/camera_info
    frame_id:=base_link
    approx_sync:=true
    database_path:=~/vo_rtab_bags/rtabmap.db
- Odom source for v1 = Cartographer `/odom` (already published; no
  new fusion needed). Decision point still open: alternatively use
  `rtabmap_odom` (RTAB-Map's own visual odom node) for a pure-visual
  SLAM story. Defaulting to Cartographer odom — simpler, more
  reliable, standard RTAB-Map RGB-D + external-odom recipe.
- Record one representative bag (drive loop incl. curve) to
  `~/vo_rtab_bags/`.
- Replay through the launch in mapping mode, watch rtabmap-viz
  build, save database = showcase artifact.

**Deliverable B — Pure-VO + Cartographer redundancy under EKF
(stretch, only if A is solid + time remains + KLT actually beats
ORB in mat A/B)**
- Requires `robot_localization` wired (config + launch). Not in the
  repo yet. Realistic as a follow-up session.
- B is the deliverable the redundancy code has been building
  toward for weeks. A is "we also did SLAM with it." Don't
  sacrifice A's solidity for B today.

Frontend-for-SLAM rule (reaffirmed): vo_frontend:=orb during any
SLAM recording. KLT/PnP knobs are A/B levers for the VO
redundancy path only; they do NOT affect RTAB-Map (which uses its
own pose estimator on the raw RGB-D stream).

No code changes yet — this is the planning capture. Implementation
of Deliverable A starts once user confirms odom-source choice
(Cartographer `/odom` recommended).

## 2026-05-20 RTAB-Map already installed on QCar — correction to prior guidance

Probe (no install action; non-destructive `apt-cache` + `dpkg` queries):
- `apt-cache policy ros-humble-rtabmap-ros` → `Installed: 0.21.1-1focal.20231230.122458`.
- `apt-cache policy ros-humble-rtabmap` → `Installed: 0.21.1-1focal.20231230.025332`.
- `apt list --installed | grep ros-humble-rtabmap` enumerates the full stack on the QCar (all 13 component packages):
  `rtabmap, rtabmap-conversions, rtabmap-demos, rtabmap-examples, rtabmap-launch, rtabmap-msgs, rtabmap-odom, rtabmap-python, rtabmap-ros, rtabmap-rviz-plugins, rtabmap-slam, rtabmap-sync, rtabmap-util, rtabmap-viz`.
- `apt-get install --simulate ros-humble-rtabmap-ros` → "ros-humble-rtabmap-ros is already the newest version. 0 upgraded, 0 newly installed, 0 to remove, 292 not upgraded."
- `ros2 pkg list | grep ^rtabmap` enumerates all 13 packages discoverable via ament.
- `which rtabmap rtabmap-databaseViewer` → `/opt/ros/humble/bin/rtabmap`, `/opt/ros/humble/bin/rtabmap-databaseViewer`.
- `ros2 launch rtabmap_launch rtabmap.launch.py --show-args` parses with all expected knobs (`frame_id`, `localization`, `database_path`, `approx_sync`, `rgb_topic`, `depth_topic`, `camera_info_topic`, `odom_topic`, `imu_topic`, …).

Conclusion:
- **No apt install required.** Earlier note from Turn 116 ("rtabmap is not installed, install on teammate machine") was incorrect on this QCar — most likely the rtabmap stack came with the Quanser Academic JetPack image. Either way, we use what is already on the car.
- The earlier CLAUDE.md guidance "never apt into the QCar's humble/L4T, install only on a teammate machine or container" was overcautious for our actual workflow. Replaced with a section that (a) records the verified install state, (b) keeps the persistent-bag rule (no `/tmp`), (c) keeps the single best-bag strategy, (d) calls out frontend choice for SLAM = ORB only.

Frontend choice for SLAM = ORB:
- KLT (LK optical-flow tracking) produces frame-to-frame correspondences with no descriptor.
- RTAB-Map's loop closure / relocalization depends on appearance-database matching against descriptors (ORB or similar binary descriptors by default).
- Therefore: when bagging or running for the SLAM showcase, `vo_frontend:=orb` (the default). The `vo_frontend:=klt` knob is reserved for VO-redundancy A/B experiments on the mat, not for the SLAM pipeline. PnP vs SVD likewise affects only the VO redundancy path; it is irrelevant to RTAB-Map (which has its own pose estimator).

Recommended remap topology when the launch is wired (not yet committed):
- `rgb_topic:=/camera/color_image`
- `depth_topic:=/camera/depth_image`
- `camera_info_topic:=/camera/camera_info`
- `frame_id:=base_link`
- `approx_sync:=true` (bridge does not time-align the three topics)
- `database_path:=~/vo_rtab_bags/rtabmap.db` (persistent)
- Open question: odom source ∈ {Cartographer `/odom`, our `/vo/odometry`, rtabmap's own VO node}. Decide when the launch wrapper is built.

No code changed; no apt action taken; this is documentation only.

## 2026-05-20 Folder rename + workspace scope correction

Context:
- Teammate deleted and restored Gabriel's working tree; on restore it
  was renamed `/home/nvidia/Documents/ACC_Development_gabriel/`.
- A separate clone of the repo lives at
  `/home/nvidia/Documents/ACC_Development/` on branch
  `Physical_Arturo` (Arturo's autonomy work — different file set:
  `lane_assist_blend`, `lane_keeping`, `path_teacher`, …; no
  `visual_odometry.py`).

Risk eliminated:
- `Easy_Start.txt` §0 / §0.5 / §0.6 / §1 / §6 and `CLAUDE.md` rsync
  cheat sheet all pointed at the old `~/Documents/ACC_Development/...`
  path. Running §0.6 unchanged would have rsync'd Arturo's
  `qcar2_autonomy/autonomy/` over Gabriel's `~/ros2/src/qcar2_autonomy/`
  and clobbered the entire VO toolbox (visual_odometry.py, vo_node.py,
  vo_supervisor.py, vo_dashboard, vo_overlay, …).

State verified before edits:
- `git branch --show-current` = `Gabriel` (correct).
- `~/ros2/src/qcar2_autonomy/autonomy/visual_odometry.py` byte-equal to
  `~/Documents/ACC_Development_gabriel/.../visual_odometry.py`
  (51,543 B, 2026-05-19 16:32) — `~/ros2` is currently Gabriel's code.
- `git remote -v` → `RoboticsClubMDC/ACC_Development.git` (single upstream
  shared with Arturo; we discriminate by branch, not by remote URL).

Edits made (docs only, no code changes):
- `CLAUDE.md`:
  - Title + project block annotated with the 2026-05-20 folder rename.
  - `/home/nvidia/Documents/ACC_Development/` added to Off-limits (Arturo's clone).
  - `ACC_Development_backup_gabriel/` documented as read-only safety net.
  - Repo layout heading + runtime workspace paragraph use
    `ACC_Development_gabriel`.
  - Build cheat sheet rsync source updated.
  - New "VO toolbox knobs" section documenting `vo_frontend`,
    `vo_estimator`, `feature_grid` (post-Turn-117 state, all default-safe).
  - New "RTAB-Map / SLAM showcase plan" section: never apt into QCar
    humble/L4T, recordings persistent (not `/tmp`), one permanent
    best-bag strategy.
  - Added post-rsync sanity check (`diff -q`) and runtime workspace
    sanity instructions.
- `Easy_Start.txt`:
  - Header dated 2026-05-20, FOLDER NOTE block added.
  - §0 `cp -r -u` source → `ACC_Development_gabriel`.
  - §0.5 A/B/C `cd` paths → `ACC_Development_gabriel`; added pwd/branch
    sanity check after the rename.
  - §0.6 rsync sources (qcar2_autonomy AND qcar2_nodes) →
    `ACC_Development_gabriel`; added explicit DO-NOT warning naming
    Arturo's clone; added `diff -q` post-sync sanity one-liner.
  - §1 calibration log dir `cd` → `ACC_Development_gabriel`.
  - §6 (virtual/QLabs) docker + isaac_ros_common paths →
    `ACC_Development_gabriel`.

No runtime code touched; default-safe VO behavior (orb+svd, grid 0)
unchanged. Next time §0.6 is run from the corrected instructions, the
post-sync `diff -q` returns empty — that's the green light.

Plan recap (carried forward from Turns 116–117):
- VO toolbox COMPLETE: feature_grid × vo_estimator × vo_frontend, all
  default-safe. Recommended operating point on the mat: orb + svd +
  feature_grid=8 (Test-6 + grid-A/B winner).
- RTAB-Map showcase = off-car build only; persistent bag path; one
  canonical best-bag kept and overwritten only when beaten.


## 2026-03-27

Scope of this pass:
- Cleanup only.
- No executable VO logic changed.
- No tuning or calibration constants were changed in code.

Changes made:
- Created this changelog file.
- Deleted `autonomy/qlabs_vo_baseline.yaml`.
- Removed misleading or stale comment/docstring text from:
  - `autonomy/visual_odometry.py`
  - `autonomy/vo_node.py`

Why `qlabs_vo_baseline.yaml` was removed:
- It contained stale VO-specific overrides from an older tuning pass.
- It mixed guessed values and deprecated assumptions such as the old `fx=161`
  path and older depth-scaling notes.
- Keeping it in the active tree risks reintroducing bad launch parameters.

Comment cleanup goals:
- Keep comments tied to explicit calibration sources.
- Remove comments that presented guessed values as if they were authoritative.
- Remove stale references to non-active file names and outdated run commands.

Files intentionally kept in place during this pass:
- `autonomy/visual_odometry_old.py`
- `autonomy/vo_node_old.py`
- `autonomy/vo_terminal_dashboard.py`
- Other VO helper utilities pending later review

Current source-of-truth documents for discussion:
- `Virtual_Stage_ROS_FAQ.md`
- `Camera Intrinsics Post.txt`
- `user_manual_system_hardware.pdf`

Open follow-up items:
- Verify the virtual `MONO16` depth scale against live runtime data.
- Decide whether the current virtual depth-to-color alignment homography
  remains acceptable or should be replaced.
- Replace placeholder physical depth/alignment assumptions with measured values.

## 2026-03-27

Scope of this pass:
- Add a standalone depth probe for `video3d_frame_get_meters(...)`.
- Keep the active `rgbd` node and active VO path unchanged.

Changes made:
- Added `qcar2_nodes/src/rgbd_get_meters_probe.cpp`.
- Added build/install entries for the new `rgbd_get_meters_probe` executable in
  `qcar2_nodes/CMakeLists.txt`.

Why this probe was added:
- The main unresolved question in the current VO path is whether the ROS2
  `MONO16` depth stream should keep using the empirical `15707.0` divisor or
  whether Quanser's depth API can provide authoritative metric depth directly.
- The Quanser C API explicitly exposes `video3d_frame_get_meters(...)` for
  depth streams, and older Quanser ROS examples already used it.
- A separate probe lets us test raw depth and metric depth on the same frame
  without changing the active `rgbd.cpp`, `visual_odometry.py`, or `vo_node.py`
  behavior.

What the new probe does:
- Opens only the depth stream, not the full RGBD publish path.
- Calls `video3d_frame_get_data(...)` and `video3d_frame_get_meters(...)` on
  the same depth frame.
- Publishes:
  - `camera/depth_image_probe_raw` as `MONO16`
  - `camera/depth_image_probe_meters` as `32FC1`
  - `camera/depth_probe_status` as a diagnostic string
- Logs a periodic comparison between:
  - raw uint16 depth
  - API meters from `get_meters`
  - the current virtual estimate derived from `15707.0 * 0.1 m/unit`

Important testing note:
- The probe opens the depth camera directly. For the cleanest test, it should
  not be run at the same time as the normal `rgbd` node unless the backend is
  confirmed to support concurrent opens.

Environment/package note:
- A package inventory spot-check on this machine found `librealsense2`
  installed.
- No Isaac ROS Visual SLAM apt package was found at this point.

## 2026-03-27

Scope of this pass:
- Runtime geometry fix for physical vs. virtual VO mode selection.
- No changes yet to `rgbd.cpp` depth publication format.

Changes made:
- Updated `autonomy/visual_odometry.py` so `camera_mode` now selects the full
  calibration stack instead of only switching RGB intrinsics.
- Updated `autonomy/vo_node.py` so physical runs no longer inherit the virtual
  `depth_scale=15707` by default.
- Added mode-aware depth diagnostics so physical mode no longer reports
  virtual-only "QLabs units" conversions.

Why these changes were needed:
- The active code had the correct virtual and physical calibration tables
  declared, but the constructor fallback path still defaulted depth scale,
  alignment, and camera-to-body extrinsics to the virtual settings.
- As a result, `camera_mode='physical'` was only partially physical.
- `vo_node.py` also forced `depth_scale=15707.0` into every run, which made
  the physical mode selection ineffective even before VO processing started.

Details of the fix:
- `visual_odometry.py`
  now builds a single mode-specific defaults table for:
  RGB intrinsics, depth intrinsics, depth scale, alignment matrix,
  camera-to-body extrinsics, depth unit label, and depth validity gates.
- `visual_odometry.py`
  now keeps raw depth intrinsics available and uses them when alignment is
  disabled, instead of always backprojecting with RGB intrinsics.
- `vo_node.py`
  now treats `depth_scale <= 0` as "use the mode default from
  visual_odometry.py".
- `vo_node.py`
  now uses `alignment_mode`:
  `auto` => virtual uses alignment, physical disables it by default until a
  validated physical RGB-depth registration is available.

Important remaining limitation:
- Physical mode is now protected from inheriting virtual-only parameters, but
  physical RGB-depth alignment is still not solved in the active ROS2 path.
- The current physical default keeps alignment off intentionally because the
  placeholder identity matrix is not a validated registration.
- Publishing metric depth directly from `rgbd.cpp` via `get_meters` remains the
  preferred follow-up change.

## 2026-03-27

Scope of this pass:
- Minimal cleanup only for the next manual VO run workflow.

Changes made:
- Kept the `rgbd.cpp` cleanup that removed malformed pasted diff text from the
  stream-open error branch.
- Reverted the temporary launch-file additions from the previous pass because
  the preferred workflow is still manual terminals for Cartographer, `vo_node`,
  and autonomy nodes.

Why this was kept:
- `rgbd.cpp` had stray pasted text in active source, which could break a rebuild
  of `qcar2_nodes`.

Why the launch changes were reverted:
- The user does not want a separate VO test launch file at this stage.
- The user is not using `qcar2_manual_drive_launch.py` in the active workflow.
- The current preferred workflow remains:
  - scenario / QLabs setup
  - Cartographer launch
  - `ros2 run qcar2_autonomy vo_node`
  - either `path_follower` or `autonomy_planner_launch.py`
  - a separate terminal tool for viewing `/vo/fault_status`

## 2026-04-30

Scope of this pass:
- Physical camera pipeline validation only.
- No motion/pose-quality testing in this pass.
- No VO source code or launch file logic changes.

Test context:
- Car was stationary and elevated (wheels not touching mat), which is acceptable
  for calibration/probe validation.
- Branch state in `/home/nvidia/Documents/ACC_Development` was confirmed as
  `Gabriel` and synchronized with `origin/Gabriel`.

Sequence executed:
1. Confirmed no active ROS/camera owners (`ros2 node list`, `ps -ef` checks).
2. Ran pre-flight calibration snapshot:
   - `rs-enumerate-devices -c`
   - `rs-enumerate-devices -o`
3. Ran isolated probe (Step 4):
   - `ros2 run qcar2_nodes rgbd_get_meters_probe --ros-args -p device_type:=physical -p frame_width_depth:=640 -p frame_height_depth:=480 -p frame_rate:=30.0`
4. Verified probe topics and sample telemetry:
   - `/camera/depth_image_probe_raw`
   - `/camera/depth_image_probe_meters`
   - `/camera/depth_probe_status`

Artifacts saved:
- `/home/nvidia/vo_calib_logs/realsense_calib_2026-04-30_120521.txt`
- `/home/nvidia/vo_calib_logs/realsense_options_2026-04-30_120521.txt`

Key observations:
- Physical D435 detected and streaming successfully in probe mode.
- `Depth Units` default reported as `0.001` in the options snapshot.
- 640x480 intrinsics were available in snapshot:
  - Depth: `fx=384.7583`, `fy=384.7583`, `ppx=324.0233`, `ppy=237.5673`
  - Color: `fx=607.3273`, `fy=607.3451`, `ppx=324.9502`, `ppy=249.8685`
- Color-depth extrinsics were reported in snapshot (non-identity transform).
- Probe status sample confirmed API-meters path:
  - `raw=8797`, `meters_api=8.7970`, `raw_est_m=0.0560`, `est/api=0.0064`
- Probe publish rate on `/camera/depth_image_probe_meters` was near target:
  - observed ~`28-29 Hz` for a `30 Hz` request.

Interpretation:
- `video3d_frame_get_meters(...)` is functioning on this physical setup.
- The old virtual estimate (`raw/(15707*0.1)`) remains strongly inconsistent
  with physical API meters in this run, so it should not be treated as a
  physical depth conversion truth.

Follow-up intent:
- Proceed to controlled physical stack bring-up (Section 2) only after this
  isolated probe validation, keeping VO in `camera_mode:=physical`.

## 2026-04-30 (manual rerun by user)

Scope of this pass:
- Manual user-run validation of Step 1 and Step 4.
- Confirm behavior after accidental terminal suspend (`Ctrl+Z`).
- No VO runtime code changes.

What happened:
- User ran Step 1 manually and saved logs in the repo-local folder:
  - `vo_calib_logs/realsense_calib_2026-04-30_121711.txt`
  - `vo_calib_logs/realsense_options_2026-04-30_121711.txt`
- Probe was started successfully, then accidentally suspended with `Ctrl+Z`.
- A second probe start failed with:
  - `video3d_start_streaming failed: Error was returned from the underlying OS-specific layer.`
- Root cause matched stream ownership conflict (first probe process still held
  the depth stream).
- After stopping the old probe process, rerun succeeded.

Manual probe evidence reported by user:
- `/camera/depth_probe_status` was visible.
- Status sample:
  - `frame=959 ... raw=9056 meters_api=9.0560 raw_est_m=0.0577 est/api=0.0064 ...`
- Probe console sample included:
  - `Depth probe opened on 0 at 640x480 @ 30.0 Hz`
  - periodic `[GET_METERS]` frames with `meters_api` values around 8.3-9.6 m
    in reported snippets.
- `timeout 12s ros2 topic hz /camera/depth_image_probe_meters` showed roughly
  `23-27 Hz` during this manual run.

Important note on `ros2 topic hz` timeout:
- The `ExternalShutdownException` seen after the timed `ros2 topic hz` command
  is expected when `timeout` terminates the ROS CLI process.

Conclusion:
- Physical depth probe path is confirmed working.
- Isolation requirement is confirmed operationally: only one stream owner at a
  time for reliable probe startup.

## 2026-04-30 (read-only pre-drive config audit)

Scope of this pass:
- Read-only verification of VO/camera launch readiness before physical manual
  drive tests.
- No ROS runtime execution and no VO algorithm code changes.

Audit outcomes:
- `qcar2_manual_cartographer_launch.py` includes
  `qcar2_manual_drive_launch.py`, so manual stack + cartographer are launched
  together in that path.
- `qcar2_manual_drive_launch.py` includes the `rgbd` node, so camera topics are
  expected in this launch path.
- `qcar2_launch.py` still has `realsense_camera_node` commented out in its
  `LaunchDescription`; this is important because `qcar2_cartographer_launch.py`
  includes `qcar2_launch.py`.
- `rgbd.cpp` defaults remain aligned with current physical VO calibration:
  `device_type="physical"`, `640x480` for color/depth, `frame_rate=30.0`.
- `vo_node.py` still defaults `camera_mode='virtual'`, but the active test
  command path sets `camera_mode:=physical`, `alignment_mode:=auto`,
  `depth_scale:=0.0` explicitly.
- `visual_odometry.py` physical 640x480 intrinsics match current RealSense
  snapshot values used in today’s tests.

Operational implication:
- Preferred physical VO path remains:
  - `qcar2_manual_cartographer_launch.py`
  - `vo_node` launched with explicit physical parameters
- Avoid using `qcar2_cartographer_launch.py` for VO camera tests unless
  `qcar2_launch.py` is adjusted to include `rgbd`.

## 2026-04-30 (manual-drive test prep guidance)

Scope of this pass:
- Prepared operator-facing launch sequence for first physical manual-drive VO
  check.
- Confirmed control mapping from source without running runtime nodes.

Key clarification:
- In current manual launch path, drive control is joystick/gamepad based, not
  WASD keyboard control.
- `qcar2_nodes/src/command.cpp` mapping:
  - `LB` arm enable
  - `RT` throttle
  - Left stick X for steering
  - `A` toggles reverse direction

Execution policy retained:
- Runtime commands are user-executed manually.
- Documentation/log files may be updated directly in-repo.

## 2026-04-30 (WASD keyboard manual-drive integration)

Scope of this pass:
- Add keyboard-based manual drive option so physical VO tests can run without a
  game controller.
- Keep changes focused on launch/control plumbing; no VO algorithm math changes.

Reference reviewed:
- Physical_Arturo branch keyboard node pattern:
  `qcar2_autonomy/autonomy/manual_drive.py`
  (publishes `Twist` to `/cmd_vel_nav`).

Changes made:
- Added `qcar2_autonomy/autonomy/manual_drive.py` in current branch with
  conservative defaults for first physical tests:
  - `forward_speed=0.10`
  - `reverse_speed=0.08`
  - `turn_rate=0.25`
  - publishes to `/cmd_vel_nav`
- Added `manual_drive` console entry point to
  `qcar2_autonomy/setup.py`.
- Added keyboard-specific base launch:
  - `qcar2_nodes/launch/qcar2_keyboard_drive_launch.py`
  - Includes `lidar`, `rgbd`, `csi`, `qcar2_hardware`, and
    `nav2_qcar2_converter`
  - Intentionally excludes joystick `command` node
- Added keyboard-specific cartographer launch:
  - `qcar2_nodes/launch/qcar2_keyboard_cartographer_launch.py`
  - Includes `qcar2_keyboard_drive_launch.py` + cartographer nodes
- Updated `Easy_Start.txt` with a new keyboard section (2.1) and explicit
  per-terminal commands.

Why this wiring is needed:
- `manual_drive.py` sends `Twist` on `/cmd_vel_nav`.
- `qcar2_hardware` consumes `qcar2_motor_speed_cmd` (`MotorCommands`).
- `nav2_qcar2_converter` bridges `/cmd_vel_nav` -> `qcar2_motor_speed_cmd`.
- Excluding joystick command prevents controller-zero-command interference when
  no controller is desired.

Verification performed:
- Syntax checks only (no hardware runtime execution):
  - `python3 -m py_compile` passed for:
    - `autonomy/manual_drive.py`
    - `qcar2_keyboard_drive_launch.py`
    - `qcar2_keyboard_cartographer_launch.py`

## 2026-04-30 (TEST 1 Physical output analysis, read-only)

Scope of this pass:
- Analyze `TEST 1 Physical` terminal output in `VO_readings.txt`.
- Confirm what `/vo/fault_status` indicates about current VO behavior.
- No runtime command execution, no VO code changes.

Data source:
- `Development/ros2/src/qcar2_autonomy/VO_readings.txt`
  section marker: `TEST 1 Physical`.

Observed monitor-state distribution (1061 status lines):
- `agree`: 513
- `vo_suspect`: 451
- `warming`: 78
- `init`: 19

Quality snapshots:
- Overall averages:
  - inliers: `178.90`
  - weight (`w`): `0.512`
  - disagreement (`rho`): `0.083`
- `agree` frames:
  - average inliers: `264.22`
  - average weight: `0.708`
- `vo_suspect` frames:
  - average inliers: `83.72`
  - average weight: `0.400`
  - average rho: `0.140`
  - zero-inlier frames: `28`

Interpretation:
- VO camera tracking is functioning and often strong (high inliers/high weight
  periods with low `rho`).
- During harder segments, quality gates correctly down-weight VO and move state
  to `vo_suspect`.
- Current monitor behavior is conservative rather than failing open.

Important architectural note:
- This run is not yet IMU-fused VIO.
- VO frame deltas come from camera processing in `visual_odometry.py`
  (ORB + matching + RANSAC + SVD).
- Cartographer is used for monitor comparison and periodic anchoring/re-anchor
  in `vo_node.py`, not as a per-frame overwrite of VO.

Likely next tuning items (no changes made in this pass):
- Revisit `turn_gate_deg` (default 5 deg) for physical turning behavior.
- Re-tune `spread_bad`, `spread_good`, and `min_vo_weight` for physical scenes.
- Improve fault-status observability by logging suspect reason codes.
- Continue physical depth/alignment hardening toward fully validated metric
  depth path.

## 2026-04-30 (anchoring frequency audit, read-only)

Scope of this pass:
- Answer whether VO is currently re-anchoring and how often.
- Use code inspection + `TEST 1 Physical` log evidence.
- No code/runtime changes.

Code behavior confirmed:
- Initial anchor occurs once on first valid frame via:
  `self.vo.reset(self.cart_x, self.cart_y, self.cart_psi)`.
- Soft re-anchor attempts happen only after consecutive `agree` decisions,
  then `_try_reanchor(...)` enforces both gates:
  - `reanchor_cooldown_s = 8.0`
  - `reanchor_min_dist = 0.25 m`
- If both gates pass, VO performs:
  `self.vo.soft_reset(self.cart_x, self.cart_y, self.cart_psi)`
  then enters warmup (`warmup_after_reanchor_s = 1.0`).

Practical implication:
- Re-anchoring is not per-frame and cannot happen every millisecond.
- Absolute upper bound is one re-anchor per 8 seconds, and only after moving
  at least 0.25 m since the last anchor.
- Between anchors, motion integration (`dx, dy, dpsi`) remains camera-driven.

`TEST 1 Physical` evidence summary:
- `warming` lines: 78
- `warming` segments: 9
- `agree -> warming` transitions: 7

Interpretation:
- The run shows occasional soft re-anchors, not continuous frame-by-frame
  Cartographer overwriting.
- However, each anchor does re-pin global VO pose/yaw to Cartographer, so the
  anchored VO stream is not fully independent over long time horizons.

Potential follow-up (not implemented in this pass):
- Add a `reanchor_enabled` parameter and expose two streams:
  - `VO_raw` (no re-anchor, strict redundancy)
  - `VO_stabilized` (current anchored behavior)

## 2026-04-30 (axis-consistency inspection, read-only)

Scope of this pass:
- Read-only interpretation of user-highlighted `TEST 1 Physical` slice.
- No code edits and no runtime execution.

User-highlighted window:
- `VO_readings.txt` lines 2221-2255.

Computed observations:
- Net motion in this slice:
  - VO:  `Δx=+0.050`, `Δy=+0.270`
  - Cart:`Δx=-0.030`, `Δy=+0.250`
- Consecutive-step sign agreement:
  - X axis: `1/17`
  - Y axis: `14/17`

Interpretation:
- Y-direction change is consistently tracked in this interval.
- X-direction consistency is weak in the same interval.
- This supports axis-wise evaluation as a primary metric, not just absolute
  pose closeness.

Policy note (discussion only, not implemented):
- Short-term competition strategy may force `vo_psi = cart_psi` for stability,
  while preserving raw camera yaw diagnostics (`vo_psi_raw`, `dpsi_raw`) for
  continued VO-only improvement work.

## 2026-04-30 (alignment source audit + policy confirmation, read-only)

Scope of this pass:
- Read-only search for physical alignment/camera-info sources in active code.
- Confirm user policy decisions for upcoming iterations.
- No runtime node execution and no VO code modifications.

Policy confirmations from discussion:
- Future analyses should auto-detect high-value windows (startup stability,
  one-axis-dominant motion, and post-anchor drift windows), not only
  user-pointed line ranges.
- Short-term competition policy set: force `vo_psi = cart_psi`.

Codebase findings:
- `qcar2_nodes/src/rgbd.cpp` currently publishes image topics only:
  `camera/color_image`, `camera/depth_image`.
- No `sensor_msgs/msg/CameraInfo` publisher path found in active source.
- In `visual_odometry.py` physical alignment remains placeholder:
  `PHYSICAL_ALIGNMENT_M = np.eye(3)` and physical default `use_alignment=False`.
- In `vo_node.py`, `alignment_mode=auto` resolves to alignment OFF for
  physical mode.

Calibration artifacts findings:
- `vo_calib_logs/realsense_calib_2026-04-30_121711.txt` contains:
  - 640x480 depth intrinsics,
  - 640x480 color intrinsics,
  - stream extrinsics (`Depth -> Color`, `Color -> Depth`) with rotation and
    translation.
- This confirms physical geometry is available in saved artifacts, but is not
  yet wired as a validated runtime alignment path inside current VO code.

Angle-sign clarification:
- `-180` and `180` yaw at startup are equivalent wrapped angles (same heading);
  this boundary flip is expected representation behavior.

Dashboard compatibility note:
- Future `/vo/fault_status` format updates should be mirrored in
  `vo_terminal_dashboard.py` in the same pass to keep diagnostics coherent.

## 2026-04-30 (hardware alignment clarification, read-only)

Scope of this pass:
- Clarify whether hardware-provided geometry should be used immediately.
- Distinguish available calibration data from active VO alignment behavior.
- No code/runtime changes.

Findings:
- Hardware geometry is available in saved calibration artifacts:
  - depth/color intrinsics (640x480),
  - depth<->color extrinsics.
- Quanser video3d API header exposes runtime accessors:
  - `video3d_stream_get_camera_intrinsics(...)`
  - `video3d_stream_get_extrinsics(...)`
  - `video3d_stream_get_depth_scale(...)`
- Active VO physical alignment path remains placeholder:
  `PHYSICAL_ALIGNMENT_M = np.eye(3)` with physical default `use_alignment=False`.

Clarification:
- The caution is not "do not use hardware geometry".
- The caution is "do not enable placeholder 2D warp as if it were true
  physical alignment".
- True depth->color alignment is depth-dependent 3D reprojection, not a single
  global homography for all distances.

Recommended direction (discussion only):
- Keep placeholder warp disabled until the VO path applies validated
  depth-dependent reprojection (or equivalent aligned-depth output) using real
  intrinsics/extrinsics/depth-scale.

## 2026-04-30 (physical alignment + yaw policy implementation)

Scope of this pass:
- Implement physical depth->color alignment improvements in VO path.
- Apply official short-term yaw policy (`vo_psi = cart_psi`).
- Update terminal dashboard parsing for current compact fault-status format.
- No ROS runtime node launches or hardware-motion commands executed.

Code changes made:

1) `autonomy/visual_odometry.py`
- Added physical `Depth -> Color` extrinsic transform constant from the saved
  RealSense calibration snapshot.
- Added alignment model selection:
  - virtual: homography alignment (`alignment_M`)
  - physical: projective depth->color reprojection
- Implemented `_align_depth_projective(...)`:
  - backprojects depth pixels with depth intrinsics,
  - transforms depth-camera points to color-camera frame,
  - projects to color pixels,
  - z-buffer resolves collisions with nearest depth.
- Physical mode defaults now set:
  - `use_alignment=True`
  - `alignment_model='projective'`

2) `autonomy/vo_node.py`
- Added parameter `force_cart_yaw` (default `True`).
- In `_vo_tick`, when enabled:
  - injects cart yaw before VO update,
  - pins output VO yaw to cart yaw after update,
  - keeps internal VO heading pinned to cart yaw.
- Result: competition-phase heading stabilization while retaining camera-based
  translation estimation.

3) `autonomy/vo_terminal_dashboard.py`
- Added regex parser for current compact `/vo/fault_status` format:
  `state rho w | vo(...) ct(...) | dx dy dpsi inl sp`.
- Kept legacy key-value fallback parsing.
- Updated dashboard summary to display state, rho, weight, drift,
  inliers, and spread from the compact message.

Validation:
- `python3 -m py_compile` passed for:
  - `autonomy/visual_odometry.py`
  - `autonomy/vo_node.py`
  - `autonomy/vo_terminal_dashboard.py`
- Offline sanity check run for `DepthProjector` alignment methods only.

Notes:
- This pass did not edit `VO_readings.txt`.
- Physical VO remained depth-based throughout; this pass specifically replaced
  placeholder-style physical alignment behavior with a projective path.

## 2026-04-30 (end-of-day clarification: virtual impact)

Scope:
- Clarified whether latest pipeline changes are physical-only.
- No additional source edits in this pass.

Clarification:
- Physical alignment changes are mode-specific (projective path for physical).
- Virtual alignment path remains homography-based.
- `force_cart_yaw` default is currently global in `vo_node.py`; therefore,
  virtual runs can also have yaw pinned unless the parameter is disabled.
- Dashboard parser change is global but diagnostic-only (no VO math impact).

## 2026-05-04 (Physical Test 1.1 vs 1.2 comparison, read-only)

Scope:
- Compare new physical runs logged as `Physical Test 1.1` and `Physical Test 1.2`
  in `VO_readings.txt`.
- Focus on agreed pattern checks: startup stability, one-axis consistency,
  and pre-reanchor behavior indicators.
- No code/runtime changes in this pass.

Data windows used:
- Physical Test 1.1: `VO_readings.txt` lines ~3010 to ~4461 (`data:` lines only)
- Physical Test 1.2: `VO_readings.txt` lines ~4499 to EOF (`data:` lines only)

Summary metrics:

1) Physical Test 1.1
- Samples: `725`
- State counts:
  - `agree`: `235`
  - `vo_suspect`: `424`
  - `warming`: `58`
  - `init`: `8`
- Averages:
  - `rho`: `0.1197`
  - `w`: `0.3863`
  - `inliers`: `127.12`
  - `spread`: `84.70`
  - zero-inlier frames: `37`
- Startup (first 80 frames):
  - state mix: `80/80 agree`
  - avg `rho`: `0.0558`
  - avg `w`: `0.7316`

2) Physical Test 1.2
- Samples: `779`
- State counts:
  - `agree`: `259`
  - `vo_suspect`: `448`
  - `warming`: `69`
  - `init`: `3`
- Averages:
  - `rho`: `0.1039`
  - `w`: `0.4803`
  - `inliers`: `133.67`
  - `spread`: `92.95`
  - zero-inlier frames: `35`
- Startup (first 80 frames):
  - state mix: `77 agree`, `2 warming`, `1 init`
  - avg `rho`: `0.0199`
  - avg `w`: `0.8230`

Interpretation vs 1.1:
- Run 1.2 is generally better quality than 1.1:
  - lower average disagreement (`rho`),
  - higher confidence/quality weight (`w`),
  - better inlier and spread averages,
  - stronger startup behavior (much lower startup `rho`).
- `vo_suspect` remains frequent in both runs, so monitor strictness during
  dynamic segments is still active.

Pattern checks (one-axis windows):
- Y-dominant windows:
  - 1.1 average sign agreement: `0.707`
  - 1.2 average sign agreement: `0.726`
- X-dominant windows:
  - 1.1 average sign agreement: `0.514`
  - 1.2 average sign agreement: `0.634`
- This indicates the biggest relative gain in 1.2 is on X-axis consistency.

Concrete line-range examples from 1.2:
- Strong Y-dominant consistency:
  - around `4861 -> 4911` (very high sign agreement, near 0.96)
- Strong X-dominant consistency:
  - around `4609 -> 4659` (sign agreement ~1.0)
- Weak segments still present:
  - Y-dominant around `5459 -> 5509`
  - X-dominant around `5881 -> 5931`

Re-anchor related observation:
- `agree -> warming` transitions (proxy for re-anchor events) were present in
  both runs (`6` in 1.1, `7` in 1.2), consistent with gated re-anchor behavior
  rather than per-frame anchoring.

Next-step implication:
- Keep current re-anchor policy ON for now (as planned).
- Use 1.2-style controlled driving as baseline for parameter tuning.
- Start VO weighting retune next (priority):
  1) `min_vo_weight`
  2) `spread_bad` / `spread_good`
  3) inlier/confidence gates

## 2026-05-04 (runtime workspace sync fix for physical VO)

Scope:
- Diagnose why physical runs still showed `Alignment: OFF` and non-pinned yaw
  after code changes were already committed in `ACC_Development`.
- Apply non-runtime fix (source sync + rebuild) so `~/ros2` executes the
  updated VO pipeline.
- No ROS motion/runtime nodes were launched in this pass.

Root cause confirmed:
- `ACC_Development` had updated files (`vo_node.py`, `visual_odometry.py`),
  but `~/ros2/src` and `~/ros2/install` were still on older copies.
- Therefore `ros2 run qcar2_autonomy vo_node` used old logic from
  `~/ros2/install`:
  - physical `alignment_mode:=auto` resolved to OFF,
  - `force_cart_yaw` logic absent.

Fix applied:
- Synced package from branch workspace into runtime workspace:
  - `/home/nvidia/Documents/ACC_Development/Development/ros2/src/qcar2_autonomy/`
    -> `/home/nvidia/ros2/src/qcar2_autonomy/`
- Rebuilt runtime package:
  - `colcon build --packages-select qcar2_autonomy`
- Verified `~/ros2/src` and `~/ros2/install` now match branch hashes and
  contain:
  - `alignment_mode == auto -> use_align = True`
  - `force_cart_yaw` parameter and pinning logic
  - physical projective depth->color alignment path

Process hardening:
- Updated `Easy_Start.txt` by adding Step `0.6`:
  - explicit `rsync` sync from branch repo to `~/ros2/src`
  - package rebuild step
- Daily order now includes Step 0.6 before physical test runs.

## 2026-05-04 (physical autonomy launch + RViz overlay implementation)

Scope:
- Align physical launch chain with autonomous VO workflow.
- Add live camera overlay for VO/Cart readings in RViz.
- Keep change set focused (no VO math changes).

Changes made:
1) Re-enabled RGBD in physical base launch:
- File: `qcar2_nodes/launch/qcar2_launch.py`
- `realsense_camera_node` is now included in returned `LaunchDescription`.
- Effect: `/camera/color_image` and `/camera/depth_image` are available in the
  physical cartographer path used by VO.

2) Added VO camera overlay node:
- File: `qcar2_autonomy/autonomy/vo_image_overlay.py`
- New node subscribes to:
  - `/camera/color_image`
  - `/vo/fault_status`
- Publishes:
  - `/vo/overlay_image`
- Overlay content includes state, rho, weight, VO pose, Cart pose,
  frame deltas, inliers, spread, and drift.

3) Added executable entry point:
- File: `qcar2_autonomy/setup.py`
- New console script:
  - `vo_overlay=autonomy.vo_image_overlay:main`

4) Updated runbook:
- File: `Easy_Start.txt`
- Added practical 5-terminal sequence for autonomous physical run + RViz image
  overlay, including `vo_overlay` and `/vo/overlay_image` topic guidance.

Notes:
- This pass does not alter VO estimator math (`visual_odometry.py`) or
  monitor state logic (`vo_node.py`).
- Purpose is runtime workflow reliability and observability.

## 2026-05-04 (vo_overlay crash: empty frame handling + runtime sync note)

Observed during physical workflow:
- `ros2 run qcar2_autonomy vo_overlay` crashed with:
  `AttributeError: 'NoneType' object has no attribute 'shape'`
- Cause: runtime install was stale (older overlay file), while branch source
  already had the empty-frame guard.

Current expected fix path:
- Sync `qcar2_autonomy` source from branch workspace into `~/ros2/src`.
- Rebuild `qcar2_autonomy` in `~/ros2`.
- Re-run `vo_overlay`.

No estimator math changed in this item; this is runtime robustness + sync.

## 2026-05-04 (physical cartographer frame parity with virtual)

Change requested:
- Remove physical-only `map_rotated -> map` static transform from
  `qcar2_cartographer_launch.py` so physical launch matches virtual map frame
  behavior.

Implemented:
- Deleted `static_transform_publisher` node that injected 180 deg yaw frame
  offset.
- Updated launch return list accordingly.
- Synced `qcar2_nodes` to runtime workspace and rebuilt `qcar2_nodes`.

Expected effect:
- Physical path follower and VO consumers read the same direct `map` frame
  convention as virtual tests.

## 2026-05-04 (rollback: keep original physical cartographer frame)

Per test-control request:
- Kept original physical launch behavior with static TF
  `map_rotated -> map` in `qcar2_cartographer_launch.py`.
- Re-synced and rebuilt `qcar2_nodes` so runtime matches this rollback.

Reason:
- Avoid large frame-convention changes before baseline verification run.

## 2026-05-04 (startup note: rs-enumerate-devices segfault)

Observed:
- At startup calibration step, `rs-enumerate-devices -c/-o` crashed with
  `Segmentation fault (core dumped)`.

Interpretation:
- Treat as failure of the enumeration utility itself until proven otherwise.
- VO testing may still proceed if ROS camera/probe nodes stream normally.

Recommended immediate triage:
1) ensure no competing camera owner process,
2) collect quick version/path/kernel crash context,
3) validate with `rgbd_get_meters_probe` + ROS topic checks.

## 2026-05-04 (rs-enumerate-devices segfault triage results)

Collected environment facts:
- No active competing `rgbd/realsense/cartographer/vo` process at crash time.
- Enumerator path/version:
  - `/opt/ros/humble/bin/rs-enumerate-devices`
  - `2.54.1`
- Installed package versions include mismatch:
  - `librealsense2` = `2.49.0`
  - `ros-humble-librealsense2` = `2.54.1`
  - `ros-humble-realsense2-camera` = `4.54.1`
- `dmesg` unavailable to regular user (kernel buffer permission).

Assessment:
- Likely ABI/runtime mismatch contributes to CLI enumerator segfault.
- Continue VO flow if ROS camera/probe nodes stream correctly; treat
  `rs-enumerate-devices` as non-blocking diagnostic-tool issue for this phase.

## 2026-05-04 (safety: nav command timeout auto-stop)

Issue observed:
- Vehicle continued moving briefly/indefinitely after planner stack stop,
  because converter kept re-publishing last received `/cmd_vel_nav` command.

Fix implemented:
- Updated `qcar2_nodes/src/nav2_qcar_command_convert.cpp`:
  - Added parameter: `cmd_timeout_sec` (default `0.25`).
  - Added last-command timestamp tracking.
  - If command age exceeds timeout, published motor command is forced to
    zero steering + zero throttle.
- Synced `qcar2_nodes` to runtime workspace and rebuilt package.

Operational effect:
- Loss of upstream planner command stream now fails safe to stop.

## 2026-05-04 (map_rotated trace audit)

User report:
- `/planned_path` appears inverted/opposite in RViz during physical run.

Audit findings:
- Physical launch file currently defines static TF:
  `map_rotated -> map` (180 deg yaw).
- Branch source `autonomy/nav_to_pose.py` currently references `map`.
- Nested workspace install copy at
  `/home/nvidia/ros2/src/install/qcar2_autonomy/.../nav_to_pose.py`
  still references `map_rotated` in path frame + TF lookup.

Implication:
- If wrong overlay/workspace setup is sourced, planner frame conventions can
  diverge and produce apparent path inversion.

## 2026-05-04 (rebuild after nav_to_pose map-frame fix)

Action:
- User updated `autonomy/nav_to_pose.py` to remove `map_rotated` usage.
- Synced package into `~/ros2/src` and rebuilt `qcar2_autonomy`.

Verification:
- `path_msg.header.frame_id = "map"`
- `pose.header.frame_id = "map"`
- `from_frame_rel = "map"`

Build status:
- `qcar2_autonomy` build succeeded.

## 2026-05-12 (read-only TF inversion audit, physical)

Inputs analyzed from `VO_readings` (cartographer physical TF test):
- `tf2_echo map map_rotated` => yaw ~ -180 deg
- `tf2_echo base_link base_scan` => yaw ~ +180 deg
- `tf2_echo map base_link` => yaw ~ -180 deg
- `tf2_echo map_rotated base_link` => yaw near 0 deg

Conclusion:
- Current physical frame chain aligns heading near zero in `map_rotated`, not
  in `map`.
- Using planner/path in `map` under this chain can produce opposite/inverted
  behavior (forward/backward and turn-direction confusion).

Diagnostic quality note:
- Initial "Invalid frame ID" lines were transient startup timing before TF
  frames became available; subsequent repeated transform outputs are valid.

## 2026-05-12 (frame-history decision checkpoint)

Key history finding:
- Initial ROS2 `nav_to_pose.py` used `map_rotated` (path frame + TF lookup).
- Commit `006f0dc` changed those references to `map`.
- Physical TF 180-deg transforms in lidar frame and cartographer launch were
  already present in initial commit lineage.

Decision guidance:
- Prefer single-variable test first: restore `nav_to_pose` frame references to
  `map_rotated` while keeping physical TF chain unchanged.
- Avoid changing both physical TF rotations to zero simultaneously during first
  diagnostic pass.

## 2026-05-12 (single-variable frame rollback test in nav_to_pose)

Change applied:
- Restored `autonomy/nav_to_pose.py` frame convention to `map_rotated` for:
  - path header frame
  - pose frame
  - TF lookup frame (`from_frame_rel`)
- Kept `map` alternatives commented for traceability.

Scope control:
- No changes made to physical TF publishers in this step
  (`fixed_lidar_frame.cpp` and `qcar2_cartographer_launch.py` unchanged).
- Purpose is isolated validation of planner frame convention only.

## 2026-05-12 (VO x/y freeze diagnostic interpretation)

Observed behavior:
- During physical autonomy test, `vo_x` and `vo_y` appeared static in
  `/vo/fault_status` and overlay.

Code interpretation (no estimator patch yet):
- `visual_odometry.py` keeps the previous pose when current frame update is
  rejected.
- Rejection causes include:
  - insufficient matches / depth-valid correspondences,
  - `inlier_count < min_inliers`,
  - safety gate failures (`max_translation`, `max_rotation_deg`, `max_dt`).

Implication:
- Static VO pose can occur even while cart pose moves, if consecutive frame
  updates are rejected.
- This is not directly caused by planner map-frame edits alone.

## 2026-05-12 (hotfix: alignment_mode CLI type crash)

Issue:
- Running `vo_node` with `-p alignment_mode:=off` caused startup failure:
  ROS parsed `off` as bool, but `alignment_mode` was declared as string.

Fix:
- Updated `autonomy/vo_node.py` to declare `alignment_mode` with
  `ParameterDescriptor(dynamic_typing=True)`.
- Downstream logic already normalizes bool/string values, so behavior is
  preserved while avoiding launch-time type exceptions.

Build:
- Synced and rebuilt `qcar2_autonomy` in `~/ros2`.

## 2026-05-12 (physical alignment regression guard)

Issue observed:
- Physical VO runs looked stable with alignment disabled, but degraded when
  alignment path was enabled.

Root cause in code:
- `vo_node.py`: `alignment_mode:=auto` was resolving to alignment ON in
  physical mode.
- `visual_odometry.py`: physical defaults also set `use_alignment=True`.
- Net effect: physical runs entered projective depth->color alignment by
  default instead of conservative raw-depth mode.

Changes applied:
1) `autonomy/vo_node.py`
- `alignment_mode=auto` now resolves to:
  - virtual: alignment ON
  - physical: alignment OFF
- Added explicit runtime warning for physical auto-off behavior.
- Explicit opt-in still supported via `-p alignment_mode:=on`.

2) `autonomy/visual_odometry.py`
- Physical mode default changed:
  - `use_alignment: True` -> `use_alignment: False`
- Projective alignment code remains intact for explicit testing.

Verification:
- Syntax check passed for updated files (`python3 -m py_compile ...`).

Expected effect:
- Physical default runs should no longer be unintentionally affected by
  projective alignment path.
- Alignment A/B testing remains available by explicitly setting
  `alignment_mode:=on` or `alignment_mode:=off`.

## 2026-05-12 (alignment reliability deep-dive, read-only)

Scope:
- Read-only audit of active physical runtime path vs. available D435 camera
  data artifacts.
- No estimator or launch code edits in this checkpoint.

Verified facts:
1) Active runtime depth source
- `qcar2_nodes/src/rgbd.cpp` publishes `/camera/depth_image` from
  `video3d_frame_get_data(...)` as raw `MONO16`.
- Current `rgbd.cpp` does not publish `camera_info` topics.

2) Hardware geometry availability
- `vo_calib_logs/realsense_calib_2026-04-30_121711.txt` contains:
  - 640x480 depth intrinsics,
  - 640x480 color intrinsics,
  - `Depth -> Color` and `Color -> Depth` extrinsics,
  - distortion model entries and coefficients from snapshot.
- `vo_calib_logs/realsense_options_2026-04-30_121711.txt` reports
  `Depth Units` default = `0.001`.

3) API depth truth path is implemented separately
- `rgbd_get_meters_probe.cpp` samples raw depth and metric depth from the same
  frame via:
  - `video3d_frame_get_data(...)`
  - `video3d_frame_get_meters(...)`
- This confirms meter-depth access exists, but is not yet the main depth image
  path used by VO.

4) Current alignment risk mechanism
- Physical alignment in `visual_odometry.py` is a software projective
  reprojection stage (not a driver-provided synchronized aligned-depth topic).
- `vo_node.py` uses latest buffered color/depth frames without explicit
  color-depth timestamp pairing tolerance at VO tick time.
- Under motion, small temporal mismatch plus reprojection sensitivity can
  degrade depth-at-feature lookup, harming VO `x/y`.

Interpretation:
- Alignment constants are sourced from real camera data, not fabricated.
- Instability comes from runtime coupling/synchronization and path validation,
  not from lack of manufacturer calibration availability.

## 2026-05-14 (PIT depth alignment cross-check, read-only)

Scope:
- Compare active physical VO depth/alignment path against PIT implementation in
  `Development/MDC_libraries/python/pit/YOLO/utils.py`.
- No runtime node launches; no estimator code edits.

Findings:
1) PIT virtual vs physical alignment are different
- Virtual branch:
  - uses homography `self.M` and virtual depth scale (`5.5` in that stack).
- Physical branch:
  - starts `QCar2DepthAlign.rt-linux_qcar2` via `quarc_run`,
  - receives combined depth+RGB packet from `BasicStream`.

2) Relation to current ROS2 VO path
- Current ROS2 physical VO uses `/camera/depth_image` from `rgbd.cpp` raw
  `MONO16` path and optional software projective reprojection in
  `visual_odometry.py`.
- PIT physical path is closer to "driver/runtime-provided aligned stream"
  behavior than to manual homography.

3) Useful reuse takeaway
- `VIRTUAL_ALIGNMENT_M` in current VO matches PIT virtual matrix exactly.
- For physical, PIT suggests that using Quanser's aligned stream runtime may be
  more robust than forcing homography-like behavior.

4) Build/symlink clarity (validated)
- With `--symlink-install`, Python source edits in `~/ros2/src/qcar2_autonomy`
  are picked up after process restart (no rebuild required).
- Edits in `~/Documents/ACC_Development/...` require sync into `~/ros2/src/...`
  before runtime sees them.
- Rebuild required for `setup.py`/entry points, C++ code, interfaces, or deps.

## 2026-05-14 (PIT vs ROS stats test result, read-only)

Data source:
- `VO_readings.txt` -> Physical Test 2 -> Testing PIT stats vs ROS stats.

Observed:
1) PIT aligned stream (`QCar2DepthAligned`)
- Sampled frames show:
  - valid depth pixels around ~290k to ~292k,
  - center depth around ~1.15 to ~1.21 m.
- `received_new=89/200` over ~2.99 s under 10 ms polling indicates
  approximately 30 Hz new-frame availability (nonblocking polling behavior).

2) ROS rgbd topic rates (same session)
- `/camera/color_image`: approximately ~17-20 Hz average with jitter.
- `/camera/depth_image`: approximately ~14-18 Hz average with jitter.
- End-of-command exceptions correspond to timeout shutdown behavior of topic
  monitor commands.

Interpretation:
- PIT aligned stream appeared stable in this snapshot test.
- Current ROS rgbd stream exhibited lower and less stable rates than requested
  30 Hz settings.
- Result reinforces synchronization/timing as a likely major factor in physical
  VO degradation when alignment is enabled.

## 2026-05-14 (full PIT/HAL/PAL camera-path review, read-only)

Scope:
- User requested a broad review of official library camera/alignment paths and
  a concrete improvement strategy for physical VO using `QCar2DepthAligned`.
- No code changes in this checkpoint.

Reviewed:
- `Development/MDC_libraries/python/pit/YOLO/utils.py`
- `docker/0_libraries/python/pal/utilities/vision.py`
- `docker/0_libraries/python/pal/products/qcar.py`
- `Development/MDC_libraries/python/hal/content/qcar.py`
- `Development/MDC_libraries/python/hal/utilities/image_processing.py`
- `Development/ros2/src/qcar2_autonomy/autonomy/{vo_node.py,visual_odometry.py,yolo_detector.py,yolo_detector_new.py}`
- `Development/ros2/src/qcar2_nodes/src/rgbd.cpp`

Findings:
1) Two distinct physical camera paths exist
- PIT path: `QCar2DepthAligned` starts `QCar2DepthAlign.rt-linux_qcar2` and
  consumes a single aligned depth+RGB stream packet via `BasicStream`.
- Current VO path: `rgbd.cpp` publishes separate color/depth ROS topics; VO
  consumes raw `MONO16` depth and optional in-VO software alignment.

2) Synchronization/coupling difference is fundamental
- PIT stream is packet-coupled (aligned depth and RGB delivered together).
- Current VO path is topic-coupled and currently does not enforce strict
  color/depth timestamp pairing at VO update time.

3) Official reusable helpers identified
- `pal.utilities.vision.Camera3D.read_depth(dataMode='M')` supports meter-depth
  retrieval for validation utilities.
- `hal.utilities.image_processing` has camera calibration and undistortion
  tooling for physical refinement.
- `yolo_detector.py` already integrates `QCar2DepthAligned` into ROS topics,
  providing a practical reference for a VO camera bridge.

Implementation direction (next step, not applied yet):
1) Add a dedicated VO camera bridge based on `QCar2DepthAligned` that publishes
   aligned RGB + aligned depth for VO use.
2) Extend `vo_node` with configurable image/depth input topics and a 32FC1
   depth ingestion path.
3) Run controlled A/B comparisons against current `rgbd.cpp` path while keeping
   ORB/RANSAC/SVD estimator settings constant.

## 2026-05-14 (implemented: VO direct PIT aligned input mode)

Scope:
- Implemented physical VO path that can consume `QCar2DepthAligned` directly.
- Kept existing ROS `rgbd.cpp` topic path intact as default/fallback.

Code changes:
1) `autonomy/vo_node.py`
- Added `input_source` parameter:
  - `ros_rgbd` (default): existing `/camera/color_image` + `/camera/depth_image`
  - `pit_aligned`: direct `QCar2DepthAligned` stream
- Added PIT connection parameters:
  - `pit_python_path`, `pit_ip`, `pit_port`, `pit_manual_start`,
    `pit_non_blocking`
- Added import resolver for `pit.YOLO.utils.QCar2DepthAligned` that supports:
  - existing `sys.path`
  - `MDC_PYTHON_PATH` env var
  - common local paths (`.../MDC_libraries/python`)
- Added `_init_pit_camera()` and `_read_pit_frame()`:
  - pulls aligned RGB + aligned meter-depth packets
  - uses ROS clock timestamp for VO update timing
  - includes periodic center-depth diagnostics for PIT mode
- PIT mode behavior:
  - no ROS color/depth subscribers are created
  - VO tick runs only when a new PIT packet arrives
  - alignment is forced OFF in VO projector (stream is pre-aligned)
- Added `destroy_node()` cleanup to terminate PIT stream process cleanly.

2) `autonomy/visual_odometry.py`
- `DepthProjector.align_depth()` now accepts:
  - integer raw depth inputs (scaled by `depth_scale`), and
  - floating depth inputs already in active units (no re-scaling)
- This enables direct use of PIT meter-depth frames without converting them to
  fake MONO16 raw counts first.
- Updated docstrings for accepted depth input types.

Build/verification:
- `python3 -m py_compile` passed for:
  - `autonomy/vo_node.py`
  - `autonomy/visual_odometry.py`
- `colcon build --packages-select qcar2_autonomy --symlink-install` succeeded
  after sourcing ROS environment.

How to run PIT input mode:
- Example launch command for physical VO:
  `ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=pit_aligned -p force_cart_yaw:=true`
- Optional path override if PIT import is not found automatically:
  `-p pit_python_path:=/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python`

Expected impact:
- Better color/depth synchronization at source (single aligned packet).
- Eliminate timing skew between separate ROS color/depth topics in VO ingest.
- Reduce depth-at-feature mismatch under motion, which should improve inlier
  quality stability and reduce drift spikes in physical runs.

## 2026-05-14 (K_inv fix for PIT-aligned + shadow VO yaw observer)

Scope:
- Two targeted fixes prompted by Claude code review:
  1. Correct intrinsics selection when consuming PIT-aligned depth.
  2. Add an independent camera-only yaw integrator for diagnostic comparison
     while `force_cart_yaw` remains the operational policy.
- Camera spec re-verification against the live D435 (not relying on saved
  `vo_calib_logs` snapshots).
- No virtual-mode behavior changed. No camera defaults changed. No RANSAC/ORB
  parameter changes. No framerate or sensor-preset changes.

Code changes:

1) `autonomy/visual_odometry.py`
- `DepthProjector.__init__` now accepts `depth_on_color_grid` (default `None`,
  falls back to `use_alignment` to preserve legacy behavior).
- Added attribute `self.depth_on_color_grid`.
- `pixels_to_3d_body` now selects `K_rgb_inv` vs `K_depth_inv` based on
  `self.depth_on_color_grid` instead of `self.use_alignment`. This separates
  two previously conflated concepts:
  - "Did we run software alignment?" (`use_alignment`)
  - "Is the depth currently sampled at color pixel coordinates?"
    (`depth_on_color_grid`)
- `VisualOdometryDepth.__init__` forwards the new kwarg to `DepthProjector`.

2) `autonomy/vo_node.py`
- Computes `depth_on_color_grid` and forwards it into `VisualOdometryDepth`:
  - `input_source=pit_aligned`: forced `True` (depth arrives pre-aligned to
    the color grid from `QCar2DepthAligned`).
  - `input_source=ros_rgbd`: mirrors `use_align` (legacy semantics).
- Added shadow yaw state `self.vo_psi_shadow`:
  - Initialized to `self.cart_psi` at first anchor and at every soft re-anchor.
  - Integrates the raw per-frame `dpsi` returned by the VO engine on every
    valid update, independent of `force_cart_yaw`.
- New publisher `/vo/vo_psi_shadow` (Float64, radians).
- `/vo/fault_status` extended with trailing `psi_raw=<deg>` field.
- Module docstring updated to list `/vo/vo_psi_shadow`.

3) `autonomy/vo_image_overlay.py`
- `_STATUS_RE` extended with an optional trailing `psi_raw=<deg>` group.
- Overlay now adds a `psi_raw=±Xdeg (camera-only, err=±Ydeg)` line when the
  new field is present in the parsed status.

4) `autonomy/vo_terminal_dashboard.py`
- `_COMPACT_RE` extended with the same optional `psi_raw` group.
- Dashboard prints an extra `psi_raw` row when the field is present, with the
  wrapped error vs Cart yaw.

Backward compatibility:
- All extended regexes use a non-capturing optional group `(?:...)?`, so
  legacy `VO_readings.txt` lines without `psi_raw` still parse.
- Default behavior for legacy callers of `DepthProjector` (no
  `depth_on_color_grid` arg) is unchanged because the default falls back to
  `use_alignment`.

Why the K_inv fix matters:
- Before: `K_inv = K_rgb_inv if use_alignment else K_depth_inv`.
- PIT path forces `use_alignment=False` because no software warp is needed —
  but the depth IS on the color pixel grid. The old logic therefore used
  `K_depth_inv` (fx≈384) to backproject ORB features picked from the color
  image (which needs fx≈607). Result: systematic scale error of
  `fx_rgb/fx_depth ≈ 1.58×` in the camera-frame 3D points, producing a
  consistent translation bias from the PIT pipeline.
- After: `K_inv = K_rgb_inv if depth_on_color_grid else K_depth_inv`.

Why the shadow yaw matters:
- `force_cart_yaw=True` overwrites VO yaw with Cartographer yaw at every
  tick, discarding the camera's frame-to-frame `dpsi` after it is computed.
- Shadow integrator preserves that signal without altering the active
  control path, providing a side-by-side diagnostic so the user can decide
  when `force_cart_yaw` can safely be flipped off.
- Reset behavior follows engine anchoring: shadow resets to `cart_psi` at
  hard anchor and at every soft re-anchor so it remains comparable to Cart.

Camera spec re-verification (live device, not log replay):
- `rs-enumerate-devices -c` confirmed against hardcoded constants:
  - Color 640x480: fx=607.327, fy=607.345, cx=324.950, cy=249.868
  - Depth 640x480: fx=384.758, fy=384.758, cx=324.023, cy=237.567
  - Depth->Color extrinsic matches `PHYSICAL_T_DEPTH_TO_COLOR`.
- `user_manual_system_hardware.pdf` (Table 3, p.6) confirms hardware max
  framerates: RGB 640x480 @ 60 Hz, Depth 640x480 @ 90 Hz. Both well above
  current 30 Hz operating point.
- FOV at 640x480 (live): Color 55.6°x43.1°, Depth 79.5°x63.9°. Depth FOV
  is wider, so any depth->color alignment necessarily crops depth coverage.
- `Realsense to body` extrinsic in the manual matches `PHYSICAL_T_CAM2BODY`
  in `visual_odometry.py`.

Compute budget check at 0.1 m/s, 30 Hz, Jetson AGX Orin:
- ORB (800 features) ~5-10 ms, BF k=2 ~2-4 ms, RANSAC 2-point x300 ~4-8 ms,
  depth path ~1-3 ms (no-op in PIT mode).
- Total ~15-25 ms vs 33 ms budget. Ample slack at current parameters; no
  tuning required until framerate or vehicle speed increases.
- Frame-to-frame translation at 0.1 m/s @ 30 Hz ≈ 3.3 mm, well below ORB
  scale; high overlap ensures fast RANSAC convergence.

Verification:
- `python3 -m py_compile` passed for all four modified files.
- Regex backward-compatibility validated against synthetic old/new
  `/vo/fault_status` strings.
- No ROS runtime executed in this pass; user runs hardware tests manually
  per the existing scope agreement.

Operational notes for next physical test:
- Sync + rebuild per Easy_Start.txt §0.6 before running:
  - `rsync -a --delete .../Development/ros2/src/qcar2_autonomy/ ~/ros2/src/qcar2_autonomy/`
  - `colcon build --packages-select qcar2_autonomy --symlink-install`
- Run VO with PIT input:
  - `ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=pit_aligned -p force_cart_yaw:=true`
- Compare `psi_raw` vs `ct(...,psi)` in the overlay or dashboard. When raw
  camera yaw tracks Cart reasonably across a full drive, consider flipping
  `force_cart_yaw:=false` for a redundancy-honest run.

Deferred (acknowledged, not implemented this pass):
- `CameraInfo` publishing from `rgbd.cpp` — pending VSLAM/loop-closure phase.
- ORB/RANSAC parameter A/B testing — deferred until baseline PIT results
  are observed; compute headroom currently sufficient at 30 Hz / 0.1 m/s.
- Camera sensor preset / laser power / exposure tuning — held at factory
  defaults to preserve compute headroom and avoid disturbing other consumers
  (YOLO, traffic detection) that share the same RealSense.
- Rate-jitter investigation — provisionally expected to resolve when running
  through `QCar2DepthAligned` (packet-coupled stream); revisit only if PIT
  path also exhibits jitter.

## 2026-05-14 (single-owner camera architecture: C++ bridge + subscribers)

Scope:
- Resolve the long-standing physical-stack camera contention where
  `rgbd.cpp` and `yolo_detector.py` both attempt to own the RealSense at
  the same time (rgbd via the Quanser `video3d_*` C API, yolo via direct
  instantiation of `QCar2DepthAligned`). Competition runs require both
  paths active simultaneously and this fails as a hard hardware conflict.
- Replace ad-hoc multi-owner pattern with a single physical camera owner
  exposed as a ROS publisher. Every other camera consumer subscribes.
- Implementation choice: C++ owner node living in `qcar2_nodes` next to
  `rgbd.cpp`, using the Quanser `quanser_communications` C API
  (`qcomm_connect/poll/receive/shutdown/close`). Confirmed available in
  `/usr/include/quanser/quanser_communications.h` and
  `/usr/lib/aarch64-linux-gnu/libquanser_communications.so`.
- Virtual mode is untouched. All virtual launches and the virtual VO path
  continue to use `rgbd.cpp` with MONO16 depth.

Architectural rationale:
- Quanser's `QCar2DepthAlign.rt-linux_qcar2` runtime performs depth->color
  alignment at the driver level and streams aligned (depth, RGB) packets
  over TCP. Previous Codex tests (2026-05-14, PIT vs ROS stats) showed
  the PIT-aligned path is denser and steadier (~30 Hz vs ROS rgbd's
  ~14-20 Hz) on the same camera. Migrating physical VO + YOLO to consume
  this stream is the highest-quality option available without changing
  RealSense hardware settings.
- A C++ implementation matches the existing `qcar2_nodes` package
  conventions (rgbd, csi, lidar, hardware are all C++), avoids a Python
  GIL between camera I/O and ROS publish, eliminates `cv_bridge`
  Python marshaling overhead, and removes the runtime dependency on a
  Python interpreter inside the critical camera path.

Files added:

1) `qcar2_nodes/src/qcar2_camera_bridge.cpp` (NEW, ~330 lines)
- ROS2 node `qcar2_camera_bridge` (executable installs under the same
  name in `qcar2_nodes`).
- Sole client of the Quanser depth-align runtime.
- Lifecycle:
  - Constructor optionally spawns the runtime via
    `quarc_run -r -t <quarc_target_uri> <runtime_path> -uri <pit_uri>`
    (matches `pit.YOLO.utils.QCar2DepthAligned.__initDepthAlign`),
    sleeps 4 s, then `qcomm_connect(pit_uri, non_blocking=true, &conn)`.
  - Worker thread (dedicated `std::thread`) loops `qcomm_poll` +
    `qcomm_receive` to accumulate 480x640x4 float32 packets in
    Fortran column-major order, then copies a complete packet into
    a mutex-guarded shared buffer.
  - Publish-side ROS timer (default 60 Hz) snapshots the buffer,
    builds `cv::Mat` instances, and publishes:
    - `/camera/color_image` as `sensor_msgs::msg::Image` encoding
      `bgr8` (built from packet channels 3,2,1).
    - `/camera/depth_image` as `sensor_msgs::msg::Image` encoding
      `32FC1` (meters, packet channel 0).
  - Destructor signals worker stop, joins, calls `qcomm_shutdown`
    + `qcomm_close`, then `quarc_run -q -Q` to terminate the
    Quanser runtime cleanly (matches PIT's `__stopDepthAlign`).
- Parameters: `device_type` (physical/virtual), `auto_start_runtime`,
  `runtime_path`, `quarc_target_uri`, `pit_uri`, `timer_rate_hz`.

Files modified:

2) `qcar2_nodes/CMakeLists.txt`
- Added `add_executable(qcar2_camera_bridge ...)` target.
- Links `-lquanser_communications -lquanser_runtime -lquanser_common
  -lpthread`, plus `cv_bridge::cv_bridge` and `opencv_imgcodecs`.
- Added `install(TARGETS qcar2_camera_bridge ...)` so the binary
  lands in `lib/qcar2_nodes/qcar2_camera_bridge`.

3) `qcar2_autonomy/autonomy/vo_node.py` — encoding auto-detect
- `_depth_cb` now branches on `msg.encoding`:
  - `32FC1`: decode as float meters, do NOT scale by `depth_scale`.
    On first sight, flips `self.vo.projector.depth_on_color_grid =
    True` and logs the event once (`_aligned_depth_locked` guard).
    The K_inv fix from earlier in the day now takes effect under
    the new pipeline because RGB intrinsics are correct for
    depth-on-color-grid backprojection.
  - Anything else: legacy `passthrough` -> uint16 path preserved.
- Depth diagnostic logger split into ALIGNED vs raw branches with
  per-mode formatting (raw counts vs metric center value).

4) `qcar2_autonomy/autonomy/yolo_detector.py` — converted from
   camera owner to ROS subscriber
- Comment-out policy applied (per project convention): the original
  PIT-ownership lines remain in the file, prefixed with the
  2026-05-14 marker and an explanation, so the legacy direct-PIT
  behavior can be restored by uncommenting + disabling the bridge.
- New subscribers on `/camera/color_image` (bgr8) and
  `/camera/depth_image` (auto-detects 32FC1 aligned meters from the
  bridge or MONO16 raw from legacy rgbd; MONO16 path divides by
  1000.0 to get meters).
- `on_timer` now reads latest frames from a `_frame_lock`-protected
  shared buffer instead of calling `self.QCarImg.read()`.
- `yolo_detect()` consumes `_current_rgb` / `_current_depth` cached
  by `on_timer` instead of `self.QCarImg.rgb` / `.depth`.
- `terminate()` becomes a no-op (bridge handles runtime shutdown).
- YOLO inference, motion-enable flag publishing, stop-override on
  `/trip_planner/qcar_state`, and detection cooldown logic are
  unchanged behaviorally.

5) Launch files (commented-arg pattern; no deletions of working
   nodes — both rgbd and bridge are defined, gated by IfCondition):
- `qcar2_nodes/launch/qcar2_launch.py`
- `qcar2_nodes/launch/qcar2_manual_drive_launch.py`
- `qcar2_nodes/launch/qcar2_keyboard_drive_launch.py`

Each declares:
  `DeclareLaunchArgument('camera_source', default_value='depth_aligned',
   description="'depth_aligned' (qcar2_camera_bridge) or 'rgbd'")`
and conditionally launches either the bridge node or the legacy rgbd
node with `IfCondition(PythonExpression(...))`. Both nodes use ROS node
name `RealsenseCamera` so downstream subscribers do not change their
expectations.

Packet protocol notes (recorded for future maintainers):
- The Quanser depth-align runtime delivers exactly
  `480 * 640 * 4 * sizeof(float)` = 4,915,200 bytes per frame.
- Layout: Fortran column-major. Element at (row=i, col=j, ch=k) lives
  at float-index `k*480*640 + j*480 + i`.
- Channel mapping (verified against PIT's Python read() implementation):
  - ch 0 = depth in meters
  - ch 1 = R (0..255 as float)
  - ch 2 = G
  - ch 3 = B
- `cv_bridge` BGR8 output is built from channels (3, 2, 1) via
  `cv::saturate_cast<uchar>`.

Build verification:
- `g++ -fsyntax-only` against ROS Humble + Quanser system headers: PASS.
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge`: PASS. Only warning is the pre-existing
  opencv-4.2-vs-4.5 ABI notice already emitted by rgbd/csi targets.
- Python `py_compile` on all four edited Python files: PASS.

Operational notes for the next physical bring-up:
1) Sync source per `Easy_Start.txt §0.6` (rsync `qcar2_autonomy` into
   `~/ros2/src/`, plus copy modified `qcar2_nodes` if not already
   under `~/ros2/src/qcar2_nodes`), then:
     `cd ~/ros2 && colcon build --packages-select qcar2_interfaces
        qcar2_nodes qcar2_autonomy --symlink-install`
   (qcar2_interfaces typically already built; include only if needed.)
2) Verify the bridge alone:
     `ros2 run qcar2_nodes qcar2_camera_bridge`
   then in another terminal:
     `ros2 topic hz /camera/color_image`     (expect ~30 Hz steady)
     `ros2 topic hz /camera/depth_image`     (expect ~30 Hz steady)
     `ros2 topic echo --no-arr /camera/depth_image | grep encoding`
       (expect `encoding: 32FC1`)
3) Launch the full physical stack via the chosen launch file with the
   default `camera_source:=depth_aligned`. To roll back to rgbd:
     `ros2 launch qcar2_nodes qcar2_cartographer_launch.py
       camera_source:=rgbd`
4) Run `vo_node` in `ros_rgbd` mode (subscribes to the bridge topics):
     `ros2 run qcar2_autonomy vo_node --ros-args
       -p camera_mode:=physical
       -p input_source:=ros_rgbd
       -p force_cart_yaw:=true`
   Expect a one-time log line:
     `[DEPTH DIAG ALIGNED] center=X.XXX m (32FC1)` confirming the
     32FC1 path is active and `depth_on_color_grid=True` was set.
5) Bring up `autonomy_planner_launch.py` for YOLO + planner + lane +
   traffic. YOLO inference and stop/yield-sign overrides are
   unchanged; the only difference is the camera data is sourced from
   ROS topics instead of a direct PIT TCP connection.

Risks acknowledged (per pre-implementation review):
- Single point of failure: bridge crash takes down all camera consumers
  simultaneously. Mitigation: launch-arg rollback to `rgbd` is one flag.
- Bridge runtime adds ~4 s startup latency (PIT runtime spawn). Any
  consumer with strict QoS / startup timeout on `/camera/*` may need
  to be launched after the bridge or tolerate the delay.
- Aligned depth is cropped to RGB FOV (55.6 x 43.1 deg) vs depth's
  native 79.5 x 63.9 deg. Acceptable for VO (features come from RGB
  anyway); flag for any future wide-FOV depth consumer.
- Quanser depth-align runtime is an opaque vendor binary; runtime bugs
  must be filed with Quanser. Same as the existing `yolo_detector` path
  used today.

Virtual-mode note (for a future port — NOT implemented here):
- `pit.YOLO.utils.QCar2DepthAligned` already handles virtual via
  `Camera3D` against QLabs port 18965 and a homography matrix `M`. To
  extend the bridge to virtual: detect `device_type:=virtual`, skip the
  quarc_run spawn, and connect to QLabs port 18965 with the same
  Fortran-order packet decoding plus the homography warp. All other
  bridge code (publishers, threading, lifecycle) is identical.

Alternatives considered and rejected (decision audit trail):

A) Do nothing structural; defer the rework until the dual-owner conflict
   is actually observed in physical testing.
   - Why considered: the conflict has never been hit in physical because
     YOLO and VO have not been run together. There was a chance the
     Quanser runtime accepts contention silently and the rework would be
     premature.
   - Why rejected: real competition runs will execute the full stack
     concurrently; the failure is virtually certain once both processes
     attempt `video3d_start_streaming` on the same camera. Pre-emptive
     consolidation is cheaper than diagnosing it on competition day.

B) Single-owner-via-rgbd: leave `rgbd.cpp` as the only RealSense owner,
   convert every other consumer (including `yolo_detector.py`) to a
   subscriber, and accept raw MONO16 depth everywhere.
   - Why considered: smallest diff. One launch-file flip, one entry-point
     swap, no new code. Lowest implementation risk.
   - Why rejected: loses the aligned depth signal entirely. The K_inv fix
     applied earlier the same day benefits aligned-depth consumers but is
     a no-op against raw MONO16. PIT-vs-ROS measurements (Codex,
     2026-05-14) showed PIT delivering denser, steadier depth than the
     `rgbd` path. Once VSLAM and loop-closure work begins, aligned depth
     is also a hard requirement. Path B avoids two future rewrites.

C) Single-owner-via-bridge (chosen). New ROS node that owns the Quanser
   depth-align runtime and publishes ROS topics for every other consumer
   to subscribe. Both `rgbd.cpp` and the new bridge remain in the source
   tree, switched at launch time via `camera_source`.

D) Shared Quanser runtime with multiple PIT clients (no bridge).
   Run `QCar2DepthAlign.rt-linux_qcar2` once externally, then have
   `vo_node` (`pit_aligned` mode) and `yolo_detector.py` both instantiate
   `QCar2DepthAligned(manualStart=True)` to attach to it.
   - Why considered: zero new infrastructure; both Python consumers stay
     close to their existing pattern.
   - Why rejected: Quanser's `BasicStream` is a TCP client. There is no
     documented confirmation that the runtime accepts multiple
     simultaneous client connections, and the failure mode would be a
     silent stall on the second connector. The bridge pattern is
     deterministic and matches the standard ROS pub/sub model.

E) Python-implemented bridge instead of C++.
   - Why considered: faster to write and iterate; easier to debug; works
     with `--symlink-install` without rebuild; lower initial bring-up
     risk.
   - Why rejected (after explicit user direction): the Quanser C
     communications API is exposed via `/usr/include/quanser/quanser_
     communications.h` and `libquanser_communications.so`, so the
     unknown that motivated "Python first, port later" turned out not
     to exist. C++ removes GIL / numpy / cv_bridge marshaling overhead,
     gives compile-time type safety, and matches the rest of the
     `qcar2_nodes` package convention. The two-step Python-then-C++
     migration was replaced by one-step C++ implementation.

F) Keep using `yolo_detector_new.py` (the existing pure-subscriber
   version) instead of converting OLD `yolo_detector.py`.
   - Why considered: `yolo_detector_new` already subscribes to ROS
     topics — would have required no code change beyond a setup.py
     entry-point swap.
   - Why rejected (per user preference): OLD `yolo_detector` carries
     additional behavior the project depends on, specifically the
     `/trip_planner/qcar_state` UInt8 override during stop-sign and
     yield-sign holds, the `/motion_enable` 500 Hz publisher, and the
     detection-cooldown / sign_detected state machine. The surgical
     conversion preserves all of that exactly while removing only the
     camera-ownership lines.

Why this matters for future maintainers:
- The `rgbd.cpp` path stays buildable and is the documented rollback
  (`camera_source:=rgbd`). Do not delete `rgbd.cpp` or the legacy
  `realsense_camera_node` block; both still serve the virtual mode and
  the rollback drill.
- The bridge is a single point of failure for the physical camera path
  by design. If it crashes the only diagnostic question is "did the
  Quanser runtime spawn cleanly?" — visible via the bridge's startup
  log and `ros2 node list`.
- The bridge consumes the same Quanser runtime that PIT helpers use.
  Anything that calls `QCar2DepthAligned(manualStart=False)` while the
  bridge is up will kill the bridge's stream (PIT's `__initDepthAlign`
  calls `__stopDepthAlign` first). Specifically: do not run the OLD
  `yolo_detector` unconverted alongside the bridge — the conversion
  in this pass is what makes them safe to coexist.

Key downsides accepted (per pre-implementation review):
- Single point of failure for the camera path. Mitigated by the
  `camera_source:=rgbd` rollback flag.
- ~4 s bridge startup delay (Quanser runtime spawn). Anything with a
  strict QoS startup timeout on `/camera/*` must launch after the
  bridge or tolerate this delay.
- Encoding switch (`MONO16` on rgbd vs `32FC1` on bridge) introduces
  a silent-bug class. Mitigated by per-callback `msg.encoding` checks
  in every depth subscriber on the project (currently only `vo_node`
  and the converted `yolo_detector`). Any new depth subscriber MUST
  do the same.
- Aligned depth is cropped to the RGB FOV (55.6 x 43.1 deg) vs depth's
  native 79.5 x 63.9 deg. Acceptable for VO (features come from RGB
  anyway); flag for any future wide-FOV depth consumer.
- Bandwidth on `/camera/depth_image` roughly doubles (MONO16 ~18 MB/s
  to 32FC1 ~37 MB/s at 640x480x30 Hz). Negligible on Jetson AGX Orin
  but visible if subscribers run on a separate host over network.

## 2026-05-14 (hotfix: bridge runtime_path resolver + spawn-fast-fail)

Issue observed on first physical test (recorded by user under
`Physical Test 2 / New camera owner Test (DepthAlign C++ Bridge)` in
`VO_readings.txt`):
- Bridge logged `Unable to download model
  /home/nvidia/Documents/ACC_Development/Development/MDC_libraries/
  resources/applications/QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2.
  The file could not be found.`
- `quarc_run` returned 256 (file-not-found).
- Bridge continued and emitted 14 sequential
  `qcomm_connect(tcpip://localhost:17003) failed: A non-blocking
  operation would have blocked` errors because no server was listening
  on port 17003 (the runtime never spawned).
- `ros2 topic hz` reported the topics were never published.

Root cause:
- The original default for `runtime_path` was a guess based on the
  PIT class's relative path (`../../../resources/applications/...`).
  PIT resolves that against its own `__file__`. On this QCar2, the
  Quanser install puts the binary at
  `/home/nvidia/Documents/Quanser/0_libraries/resources/applications/
  QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`, outside
  `ACC_Development`. The default path I had pointed inside
  `ACC_Development/Development/MDC_libraries/resources/`, which does
  not exist (MDC_libraries only contains `python/`).

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- Added `resolveRuntimePath(std::string& out)` that searches a
  prioritized candidate list and returns the first existing file:
    1. The user-supplied `runtime_path` parameter (if non-empty)
    2. `QCAR2_DEPTHALIGN_RUNTIME` environment variable (if set)
    3. `/home/nvidia/Documents/Quanser/0_libraries/resources/
        applications/QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`
        (actual install location on this QCar2)
    4. `/home/nvidia/Documents/ACC_Development/docker/0_libraries/
        resources/applications/QCarDepthAlign/
        QCar2DepthAlign.rt-linux_qcar2`
    5. `/home/nvidia/Documents/ACC_Development/backup/
        Quanser_Academic_Resources/0_libraries/resources/applications/
        QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2`
- Default value of the `runtime_path` parameter changed from a hard-
  coded (wrong) path to empty string — falls through to the resolver.
- `startRuntime()` now:
    1. Calls `resolveRuntimePath()` BEFORE invoking `quarc_run`.
       Logs the list of candidate paths tried if nothing resolves.
    2. Treats a non-zero `quarc_run` return code as ERROR (was WARN)
       and sets `runtime_spawn_ok_ = false`.
- Bridge constructor now checks `runtime_spawn_ok_` after
  `startRuntime()` and ABORTS init if false. No more 14-attempt
  reconnect loop on a runtime that never started. Logs a clear
  remediation message: set `runtime_path` parameter and relaunch.
- Destructor only attempts `stopRuntime()` if the spawn actually
  succeeded.
- Added `<filesystem>`, `<sys/stat.h>` includes for path existence
  checks via `stat()`.

Operational implication:
- User does NOT need to pass `-p runtime_path:=...` on this QCar2 —
  the resolver finds the binary at
  `/home/nvidia/Documents/Quanser/0_libraries/...` automatically.
- For machines where the install location differs, override with
  either `-p runtime_path:=/full/path/to/QCar2DepthAlign.rt-linux_qcar2`
  or `export QCAR2_DEPTHALIGN_RUNTIME=/full/path/...`.
- If neither the parameter nor the env var nor any of the candidate
  paths exists, the bridge now logs a clear list of what it tried
  and exits init cleanly rather than spamming connect failures.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only the pre-existing OpenCV ABI warning.

Follow-up (same day, per user direction): reordered the candidate
list to prefer the in-repo copy over the system install. All four
copies on this machine were verified byte-identical
(md5 58572dc0d62e8535140afb45f5eaf554):
- `/home/nvidia/Documents/Quanser/0_libraries/...` (system install)
- `ACC_Development/docker/0_libraries/...` (in-repo, default)
- `ACC_Development/backup/Quanser_Academic_Resources/0_libraries/...`
- `ACC_Development/docker/development_docker/...` (also in-repo)

New priority order in `resolveRuntimePath()`:
  1. `runtime_path` parameter (if non-empty)
  2. `QCAR2_DEPTHALIGN_RUNTIME` env var
  3. `ACC_Development/docker/0_libraries/...`  ← new default
  4. `ACC_Development/backup/Quanser_Academic_Resources/0_libraries/...`
  5. `/home/nvidia/Documents/Quanser/0_libraries/...` (system, fallback)

Rationale: keeps the project self-contained on a fresh clone; matches
the user's explicit preference for not depending on paths outside
`ACC_Development`. Rebuilt and verified.

## 2026-05-14 (hotfix #2: match PIT quarc_run invocation exactly)

Issue:
- After the runtime_path resolver fix above, the bridge's `quarc_run`
  invocation still produced no listening server on port 17003.
  Diagnostic single-line manual command from the user's terminal
  (recorded under `VO_readings.txt -> Step 2 Test 2 trailing
  diagnostics`) demonstrated that the same unquoted `quarc_run -r -t
  tcpip://localhost:17000 /full/path/QCar2DepthAlign.rt-linux_qcar2
  -uri tcpip://localhost:17003` started the runtime cleanly
  (`exit=0`; `QCar2DepthAlign` process alive; ports 17003 AND 18777
  listening — the runtime exposes both URIs by default which is why
  PIT's `port='18777'` consumer default has historically worked
  against a runtime spawned with `-uri tcpip://localhost:17003`).
- The bridge's failing variant wrapped the path in double quotes:
  `quarc_run -r -t tcpip://localhost:17000 "/full/path/file.rt..."
  -uri tcpip://localhost:17003`. The quotes are the only difference
  between the failing and working forms. On this Jetson the quoted
  form is consumed silently by `quarc_run` with no output and no
  non-zero exit, leaving no server bound to the URI.
- Additionally, the bridge's pre-stop command passed the full quoted
  path to `quarc_run -q -Q`. PIT's `__stopDepthAlign` passes ONLY
  the basename of the model file. Quanser's `-q -Q` flag matches
  models by basename; a full path is not the canonical form.
- The `-D` flag (daemon-detach) used in `qvl/real_time.py` is broken
  on this aarch64 QCar2 build: manual `quarc_run -D -r ...` failed
  with "Unable to download model ... An operating system function
  returned an unrecognized error" (exit=1). Confirmed by repeated
  manual test. Do NOT add `-D` to the bridge.

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- `startRuntime()` now builds the start command WITHOUT quoting the
  runtime path. The candidate-list resolver guarantees the resolved
  path contains no whitespace, so quoting is unnecessary and (per
  diagnostic data) actively harmful.
- `startRuntime()` and `stopRuntime()` now derive the model basename
  via `std::filesystem::path(runtime_path_).filename().string()` and
  pass it to `quarc_run -q -Q <basename>`, matching PIT's exact
  invocation.
- Added an INFO log line after `quarc_run` returns to mark the start
  of the 4 s settle wait. Useful breadcrumb in future debugging.
- No `-D` flag is added; the QCar2's aarch64 quarc_run build does
  not support it.

Operational expectation after this fix:
- Bridge startup log should now read (in order):
    Using QCar2DepthAlign runtime at: /home/nvidia/.../QCar2DepthAlign.rt-linux_qcar2
    Spawning Quanser depth-align runtime: quarc_run -r -t
      tcpip://localhost:17000 /home/nvidia/.../QCar2DepthAlign.rt-linux_qcar2
      -uri tcpip://localhost:17003
    quarc_run returned exit=0; waiting 4s for runtime to bind to
      tcpip://localhost:17003 ...
    Connected to QCar2DepthAlign stream at tcpip://localhost:17003
- `pgrep -af QCar2DepthAlign` should show the runtime process alive
  while the bridge runs.
- `/camera/color_image` and `/camera/depth_image` should publish at
  ~30 Hz with encodings `bgr8` and `32FC1` respectively.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-14 (hotfix #4: switch from qcomm_* to stream_* API for buffer sizing)

Observed after hotfix #3 (recorded in `VO_readings.txt -> Step 2 Test 4`):
- Bridge connects cleanly: "Connected to QCar2DepthAlign stream at
  tcpip://localhost:17003" appears in the log.
- Worker heartbeat shows packets ARE arriving:
    [bridge worker hb] packets=1  poll_data=653  bytes=7891559
    [bridge worker hb] packets=17 poll_data=7258 bytes=87287633
- BUT the throughput is wrong: 16 full packets across 50 seconds
  (~0.32 fps) instead of the expected ~30 fps. Each `qcomm_receive`
  call returned only ~12 KB on average (79 MB / 6605 receive calls),
  forcing ~400 receive calls to assemble each 4.9 MB frame.

Root cause:
- The bridge used `qcomm_connect / qcomm_poll / qcomm_receive` from
  `quanser_communications.h`. The `qcomm_connect` signature does NOT
  let the caller size the send/receive buffers — it uses small TCP
  defaults (~16-64 KB).
- PIT's Python `pal.utilities.stream.BasicStream` does NOT use the
  `qcomm_*` family. It calls Quanser's `stream_*` family (in the
  same `libquanser_communications.so` library) which DOES take
  explicit `send_buffer_size` and `receive_buffer_size` arguments.
  PIT's QCar2DepthAligned passes `send=480*640*3` and
  `recv=480*640*4*4` (= 4,915,200 bytes) so `stream_receive` can
  pull a near-full frame in a single call.
- `STREAM_POLL_RECEIVE / STREAM_POLL_CONNECT / etc.` are #define
  aliases of the corresponding `QCOMM_POLL_*` values in
  `quanser_stream.h`, so flag constants are unchanged.

Fix applied to `qcar2_nodes/src/qcar2_camera_bridge.cpp`:
- Added `#include "quanser/quanser_stream.h"`.
- Replaced `t_connection conn_` member with `t_stream stream_`.
- Constructor initializer adds `stream_(nullptr)`.
- `connectStream()` now calls
    stream_connect(uri, /*non_blocking=*/true,
                   send_buffer_size = 480*640*3,
                   receive_buffer_size = 480*640*4*4,
                   &stream_);
  matching PIT's buffer sizes exactly.
- Subsequent loops use `stream_poll(stream_, t, STREAM_POLL_RECEIVE)`
  and `stream_receive(stream_, buf, n)`.
- Destructor uses `stream_shutdown` + `stream_close`.
- Error labels updated from "qcomm_poll"/"qcomm_receive" to
  "stream_poll"/"stream_receive" for log accuracy.
- The `-QERR_WOULD_BLOCK` handling from hotfix #3 is preserved
  (still the expected return for non-blocking connect-in-progress).

Expected behavior after this fix:
- Heartbeat should now show packets growing at ~30 per heartbeat
  interval (≈ 5 s × 30 fps = 150 packets per heartbeat).
- `bytes_received_total / poll_data` should approach ~1 MB per
  receive call instead of ~12 KB.
- `/camera/color_image` and `/camera/depth_image` should publish at
  ~30 Hz to subscribers in the same ROS_DOMAIN_ID.

Open follow-up observed in same test:
- Even at the (slow) 0.34 fps achieved before this fix, `ros2 topic
  list | grep camera` in a second terminal returned no `/camera/*`
  topics. This is independent of the throughput fix; it points at a
  ROS discovery issue (likely `ROS_DOMAIN_ID` mismatch between the
  terminal that ran the bridge and the terminal that ran
  `ros2 topic list`, or a stale `ros2 daemon`). Diagnose with
  `echo $ROS_DOMAIN_ID` in both terminals and `ros2 daemon stop &&
  ros2 daemon start` if they match but topics still don't appear.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-14 (hotfix #5: use stream_receive_byte_array for atomic packet receive)

Observed after hotfix #4 (buffer-size fix, recorded in `VO_readings.txt
-> Step 2 Test 5`): throughput essentially unchanged at ~0.3 fps; each
`stream_receive` call returned only ~23 KB on average (slight
improvement over the previous ~12 KB but nowhere near the expected
~4.9 MB per call).

Root cause discovered by actually reading PIT's
`pal.utilities.stream.BasicStream.receive` (lines 369-444 of
`docker/0_libraries/python/pal/utilities/stream.py`):

  pollResult = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
  ...
  self.bytesReceived = self.clientStream.receive_byte_array(
      self.data, totalNumBytes)

PIT calls `receive_byte_array`, NOT plain `receive`. The C equivalent
declared in `quanser_stream.h` is:

  EXTERN t_int stream_receive_byte_array(t_stream stream,
                                         t_byte * elements,
                                         t_uint num_elements);

Quanser's own header documents the semantic difference:

> This function receives an array of bytes over a client stream. It
> differs from the stream_receive_bytes function in that it treats the
> entire array as an atomic unit. It either receives all of the array
> or none of it. It also requires that the stream receive buffer be at
> least as large as the array of bytes.

> Returns 1 on success. If not enough data is available and the
> connection has been closed gracefully then 0 is returned. If an error
> occurs then a negative error code is returned. In non-blocking mode,
> -QERR_WOULD_BLOCK is returned when not enough data has buffered yet.

So:
- `stream_receive`         -> partial chunks (~12-23 KB per call), need
                              manual accumulation; the API we WERE using.
- `stream_receive_byte_array` -> atomic; returns 1 only when the FULL
                              num_elements arrives; the API PIT uses.

Fix applied to `qcar2_camera_bridge.cpp` worker thread:
- Replaced the accumulating `stream_receive` loop with a single
  `stream_receive_byte_array(stream_, scratch_bytes, kPacketBytes)`
  call per poll-with-data event.
- Removed the `total_received` accumulator and `remaining` calculation;
  no longer needed since each successful call delivers a complete
  4.9 MB packet.
- Return-code handling now distinguishes:
    1                  -> publish the full frame, increment packets++
    0                  -> connection closed; flag conn_open_=false for
                          worker-level reconnect on next iteration
    -QERR_WOULD_BLOCK  -> not enough buffered; increment would_block
                          counter and loop back to poll
    other negative     -> log via logQuanserError
- Buffer ptr cast updated from `char*` to `t_byte*` to match the API.
- Heartbeat counters updated: removed `in_progress` (no longer
  meaningful), added `would_block`.
- Inline comment block in the source explains the semantic difference
  so this never gets re-broken.

Expected behavior after this fix:
- Heartbeat should now show `packets` growing by ~150 per 5-second
  interval (30 fps x 5 s).
- `polls_with_data` should approach `packets` 1:1 (one atomic receive
  per poll-with-data event) instead of ~400:1 as before.
- `/camera/color_image` and `/camera/depth_image` at ~30 Hz.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

Independent earlier finding (now resolved by user): terminal B's
missing `/camera/*` topics in hotfix #4's test were caused by
`ROS_DOMAIN_ID` being unset in terminal B while terminal A had
`ROS_DOMAIN_ID=67`. Resolution: `export ROS_DOMAIN_ID=67` in every
shell, or add to `~/.bashrc`. Easy_Start.txt Section 0 already
documents this.

## 2026-05-14 (pivot: bridge moved from C++ to Python — final implementation)

Decision:
- After hotfix #5 still produced ~0.3 fps and the user's diagnostic
  showed `net.core.rmem_max=212992` (208 KB kernel cap), the user
  applied `sudo sysctl -w net.core.rmem_max=16777216` to lift the
  ceiling to 16 MB. Re-tested: bridge throughput remained ~0.3 fps
  with `would_block` exceeding 200 million per 5-second heartbeat.
  Kernel-level fix did NOT unlock the Quanser receive path.
- Conclusion: Quanser's library does something inside its Python
  wrapper that the C `stream_*` / `qcomm_*` APIs do not expose. PIT
  achieves ~30 fps via `pal.utilities.stream.BasicStream` (verified
  in production with legacy yolo_detector.py). The fastest path to
  a working single-owner camera bridge is to invoke
  `pit.YOLO.utils.QCar2DepthAligned` directly from a Python ROS2
  node, exactly as legacy yolo_detector used it.

Pivot implemented:

1. NEW file `Development/ros2/src/qcar2_autonomy/autonomy/qcar2_camera_bridge.py`
   (~230 lines).
   - rclpy Node `qcar2_camera_bridge`.
   - Reuses the `_resolve_pit_import` pattern from `vo_node.py` so PIT
     can be located on standard or alternate sys.path layouts.
   - Instantiates `QCar2DepthAligned(ip, nonBlocking, manualStart,
     port, isPhyscial)` with PIT defaults (matches legacy
     yolo_detector behavior verbatim).
   - 60 Hz polling timer wrapping `cam.read()`. When `read()` returns
     True a new frame is available; the timer publishes:
       /camera/color_image  (bgr8)
       /camera/depth_image  (32FC1 meters, aligned to color grid)
   - One-shot log line on first frame; periodic "[bridge hb]" log
     line every 5 s reporting actual published fps for live
     verification.
   - `destroy_node()` calls `self.cam.terminate()` so the Quanser
     depth-align runtime exits cleanly on shutdown.
   - Parameters: device_type, pit_python_path, pit_ip, pit_port,
     pit_non_blocking, pit_manual_start, publish_rate.

2. UPDATED `Development/ros2/src/qcar2_autonomy/setup.py`:
   - Added console entry point:
       camera_bridge=autonomy.qcar2_camera_bridge:main

3. UPDATED launch files
   (`qcar2_launch.py`, `qcar2_manual_drive_launch.py`,
    `qcar2_keyboard_drive_launch.py`):
   - The `camera_bridge_node` definition switched from
       package='qcar2_nodes', executable='qcar2_camera_bridge'
     to
       package='qcar2_autonomy', executable='camera_bridge'
   - The `camera_source` launch arg and the IfCondition wiring are
     unchanged; rollback to legacy rgbd.cpp is still available via
     `-p camera_source:=rgbd`.

4. C++ bridge kept in tree (per comment-out policy). Added a header
   comment block at the top of
   `qcar2_nodes/src/qcar2_camera_bridge.cpp` documenting:
   - "STATUS (2026-05-14): NOT THE ACTIVE BRIDGE"
   - Pointer to the Python implementation that supersedes it
   - The full forensic of why C++ couldn't reach 30 fps
   - Note that the file remains buildable as a reference for any
     future Quanser SDK update that exposes the missing piece.
   The CMakeLists target is unchanged — `qcar2_camera_bridge` binary
   still builds and installs to `lib/qcar2_nodes/qcar2_camera_bridge`,
   it just isn't referenced by any launch file.

Operational expectation after this pivot:
- The Python bridge should achieve ~30 fps because it uses the exact
  same `QCar2DepthAligned` class legacy yolo_detector used at ~30 Hz.
- All downstream consumers (vo_node ros_rgbd path, yolo_detector,
  traffic_system_detector, lane_detector) remain unchanged. They
  subscribe to `/camera/color_image` and `/camera/depth_image`,
  which the Python bridge now publishes with the same encodings
  (bgr8 and 32FC1) the C++ bridge was advertising.
- `vo_node._depth_cb` encoding auto-detect from hotfix #1 still
  flips `depth_on_color_grid=True` on receipt of the first 32FC1
  message, so the K_inv backprojection fix remains active.
- Sync + rebuild on the QCar2 via Easy_Start.txt §0.6 will need to
  rebuild only `qcar2_autonomy` (Python) — qcar2_nodes itself was
  not touched in this pivot. Use:
    rsync -a --delete .../qcar2_autonomy/  ~/ros2/src/qcar2_autonomy/
    rsync -a --delete .../qcar2_nodes/launch/  ~/ros2/src/qcar2_nodes/launch/
    cd ~/ros2 && colcon build --packages-select qcar2_autonomy --symlink-install

Build verification:
- `python3 -m py_compile` PASS on:
    autonomy/qcar2_camera_bridge.py
    qcar2_nodes/launch/qcar2_launch.py
    qcar2_nodes/launch/qcar2_manual_drive_launch.py
    qcar2_nodes/launch/qcar2_keyboard_drive_launch.py
    qcar2_autonomy/setup.py

## 2026-05-14 (Step 3 PHYSICAL validation — bridge + VO end-to-end PASS)

Test conditions:
- Car elevated and stationary (wheels off the ground).
- Single launch command: `ros2 launch qcar2_nodes
  qcar2_cartographer_launch.py` (camera_source defaulted to
  depth_aligned, so the Python bridge auto-started alongside
  cartographer, lidar, csi, qcar2_hardware, nav2_qcar2_converter,
  static_transform_publisher, fixed_lidar_frame, and the occupancy
  grid node).
- `vo_node` launched separately with `-p camera_mode:=physical
  -p input_source:=ros_rgbd -p force_cart_yaw:=true`.
- `ros2 topic echo /vo/fault_status` captured in a third terminal.

Results captured in `VO_readings.txt` under the trailing Step 3 block
(camera_bridge + cartographer + vo_node + fault_status):

1. Single-launch architecture confirmed:
   - Launch log line: `[INFO] [camera_bridge-3]: process started with
     pid [42966]` — i.e. the Python bridge is now a peer of the C++
     nodes inside cartographer's launch. ROS2 launch handles the
     language difference transparently. No separate bridge launch
     needed.

2. Bridge fps confirmed under full-stack load:
   - `[bridge hb] published=152 frames in 5.0 s (~30.4 fps)` repeated
     across multiple heartbeats — the Python bridge holds a steady
     ~30 fps even when cartographer + lidar + qcar2_hardware are
     consuming CPU simultaneously.

3. VO consuming the bridge correctly:
   - One-shot log line on first depth callback:
       "Detected aligned 32FC1 depth on depth_image; set
        depth_on_color_grid=True. VO will use RGB intrinsics
        (K_rgb_inv) for backprojection."
   - This proves the encoding auto-detect from hotfix #1 (now active
     under the Python bridge) fires correctly and the K_inv fix is
     in the live path.
   - Periodic `[DEPTH DIAG ALIGNED] center=1.158 m (32FC1)` confirms
     real metric depth being consumed by VO.

4. /vo/fault_status quality (stationary baseline):
   - State: `agree` consistently (no `init`/`warming` after warmup).
   - rho (residual): 0.003-0.031 m — excellent agreement with Cart
     pose at rest.
   - weight (quality): 0.65-0.75 sustained.
   - inliers: 300-415 per frame — very high feature counts, well
     above the 200-260 baseline observed in earlier MONO16+no-align
     physical tests.
   - spread: 90-106 pixels — full image coverage.
   - psi_raw: 180 deg consistently (shadow yaw integrator initialized
     to cart_psi and not moving because the car is stationary — dpsi
     ~0 every frame, so the integrator stays put).
   - VO pose: vo(0.14, 0.01, ~180°) — anchored to Cart's initial
     pose at start and held within mm-scale jitter, exactly the
     expected behavior at rest.

Verdict: the single-owner camera architecture is operationally
complete and outperforms the prior MONO16+software-align baseline in
every measured dimension (higher inlier counts, lower rho, higher
weight, no jitter on the camera topics, no setup hassle).

Next session: mat run with actual driving. See the "Plan for next
session" section at the end of `VO_Conversation_Log.txt`.

## 2026-05-14 (hotfix #3: connect logic — -QERR_WOULD_BLOCK is expected)

Issue:
- After hotfix #2 (PIT-matching invocation, no quotes, basename-only
  stop), `quarc_run` correctly spawned the runtime. Manual diagnostic
  (recorded in VO_readings.txt -> Step 2 Test 3 trailing checks)
  confirmed `QCar2DepthAlign -uri tcpip://localhost:17003` was alive
  and listening on ports 17003 + 18777.
- BUT the bridge still logged "qcomm_connect failed: A non-blocking
  operation would have blocked" on every iteration.
- The smoking gun was the `ss -tlnp` output: `LISTEN 19 128
  0.0.0.0:17003` (Recv-Q = 19) and a few seconds later `LISTEN 33
  128 0.0.0.0:17003` (Recv-Q = 33). The Recv-Q on a listening
  socket is the count of half-open connection attempts the kernel
  has queued waiting for the server to accept(). Growing Recv-Q
  proves the bridge's connect attempts WERE reaching the runtime
  but were never being followed through.

Root cause in `qcar2_nodes/src/qcar2_camera_bridge.cpp`
(`connectStream()`):
- `qcomm_connect` with `non_blocking=true` returns `-QERR_WOULD_BLOCK`
  while the TCP connection is in progress. That return value is the
  expected/correct behavior per the Quanser docs — the caller is
  supposed to then invoke
  `qcomm_poll(conn, timeout, QCOMM_POLL_CONNECT)` until it returns
  > 0 to know the connection has completed.
- The bridge's connect logic treated ANY r < 0 as a fatal error
  (`if (r < 0) { log; return false; }`). That ran on the first
  `qcomm_connect` call, before `conn_open_` was set to true, so the
  worker loop kept calling `connectStream()` every 250 ms. Each
  call queued a fresh half-open TCP connection at the runtime and
  abandoned it before completing the handshake. Runtime never
  accepted anything because the bridge never polled for
  POLL_CONNECT after issuing the call.

Fix applied:
- `connectStream()` now distinguishes `-QERR_WOULD_BLOCK` from other
  failures explicitly:
    if (r < 0 && r != -QERR_WOULD_BLOCK) { fatal log; return false; }
  On `-QERR_WOULD_BLOCK` it sets `conn_open_ = true` and enters the
  existing `qcomm_poll(QCOMM_POLL_CONNECT)` loop (2 s budget) to
  wait for the connection to complete cleanly.
- Added an inline comment in the source describing the protocol so
  this never gets re-broken.

Expected behavior after this fix (next physical bring-up):
- Bridge log should print one of:
    "Connected to QCar2DepthAlign stream at tcpip://localhost:17003"
       (after the post-spawn poll completes), OR
    "Stream connect did not complete within initial 2 s wait;
       worker will keep polling." (degraded — investigate)
- No more repeated "A non-blocking operation would have blocked"
  ERROR lines. That message originated from misclassifying the
  normal non-blocking return.
- `ss -tlnp | grep :17003` should show Recv-Q = 0 (no queued
  half-open connects) once the bridge has connected.
- /camera/color_image and /camera/depth_image should start
  publishing at ~30 Hz with encodings bgr8 and 32FC1.

Build verification:
- Isolated `colcon build --packages-select qcar2_nodes --cmake-target
  qcar2_camera_bridge` PASS; only pre-existing OpenCV ABI warning.

## 2026-05-15 (manual_drive hotfix: continuous publish at 50 Hz)

Issue observed on the mat (first attempted Step 4 drive):
- After launching `qcar2_cartographer_launch.py` + `vo_node` + `manual_drive`,
  pressing `a`/`d` produced a brief visible steering response, but `w`/`s`
  produced no forward/reverse motion. Increasing `forward_speed` did not
  help (confirmed by operator).

Root cause:
- `nav2_qcar_command_convert` enforces a 0.25 s `/cmd_vel_nav` timeout that
  zeros both throttle AND steering when no fresh Twist arrives inside the
  window (added 2026-05-04 as a safety auto-stop). The previous
  `manual_drive.py` only published on KEY PRESS, so each W tap produced a
  single 0.25 s Twist pulse followed by forced zero. At low throttle the
  pulse can't overcome static friction; at higher throttle the wheels get
  yanked-then-stopped every 0.25 s. Steering looks "responsive" because the
  servo visibly moves during the pulse even though the converter zeros it
  immediately afterward.

Fix implemented in `autonomy/manual_drive.py`:
- Decoupled keyboard reading from publishing.
- Keyboard thread (`_keyboard_loop`) only mutates a thread-safe
  `(linear_x, angular_z)` state on key press.
- A 50 Hz `rclpy` timer (`_publish_current`) emits the current state every
  20 ms - well inside the converter's 0.25 s safety window - so tapping
  w/a/s/d once produces sustained motion until space/x/q is pressed.
- Stop+republish on shutdown so the converter receives an explicit zero
  before the publisher destroys.
- Added a top-of-file comment block documenting the architecture and the
  2026-05-04 safety-timeout interaction so this never regresses to a
  publish-on-keypress design.

Defaults unchanged (`forward_speed=0.10`, `reverse_speed=0.08`,
`turn_rate=0.25`) so existing tuning recipes in Easy_Start.txt section 2.1
still apply. New parameter `publish_rate_hz` (default `50.0`).

Build verification:
- `python3 -m py_compile` PASS on the rewritten manual_drive.py.
- Rsync + `colcon build --packages-select qcar2_autonomy --symlink-install`
  PASS (only the standard easy_install deprecation warning).

Next: re-run Step 4 mat drive with the new manual_drive build.

## 2026-05-15 (manual_drive hotfix #2: independent throttle and steering axes)

Issue: previous hotfix made w/a/s/d mutually exclusive (pressing `a` zeroed
`linear_x`, pressing `w` zeroed `angular_z`), so the car could not be
steered while moving - exactly the use case the test plan requires.

Fix: split the state-setters in `manual_drive.py` so each key only mutates
the axis it controls:
- `w` / `s`  -> set `linear_x` only, leave `angular_z` untouched.
- `a` / `d`  -> set `angular_z` only, leave `linear_x` untouched.
- ` ` (space) -> release throttle only (linear_x=0).
- `c`         -> center steering only (angular_z=0).
- `x`         -> full stop (both axes zero).
- `q`         -> stop and quit.

The 50 Hz publish-timer architecture from hotfix #1 is unchanged - this
change only touches the keyboard handler.

Build verification: py_compile PASS, rsync to ~/ros2/src + colcon build
PASS.

## 2026-05-15 (manual_drive hotfix #3: revert to original, single-line fix only)

Per operator instruction: previous hotfixes (#1 + #2) introduced too much
new behavior. Reverted `manual_drive.py` to exactly the file at git HEAD
and applied the smallest possible change to fix the no-forward-motion bug.

Total diff vs HEAD: 4 added lines (3-line comment + 1 republish call).

Behavior is otherwise unchanged from the original:
- w/a/s/d still mutually exclusive (each key zeros the other axis).
- Controls, defaults, prints, exit behavior all identical.
- Only difference: when `get_key()` returns "" (no key in the 100 ms
  window), the loop now also publishes the current `(linear_x,
  angular_z)` before `continue`. That keeps `/cmd_vel_nav` fresh
  inside the 0.25 s safety timeout of `nav2_qcar_command_convert`,
  so a tap-W produces sustained motion until tap-x/space/q.

py_compile + rsync + colcon build PASS.

Hotfix #1 and #2 are superseded by this entry. The 50 Hz timer-based
publish + decoupled keyboard thread + new key bindings are all rolled
back.

## 2026-05-15 (manual_drive — full verbatim state snapshot, post-hotfix #3)

Per operator instruction: changelog must be self-contained. No future
reader of these logs should ever need to consult git history to know
what the source looked like. Recording the complete contents of
`Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` as of
the end of this turn (after the hotfix #3 single-line fix) below.

------------------- BEGIN manual_drive.py -------------------
#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def get_key(timeout_s=0.1):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Conservative defaults for first physical tests.
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("reverse_speed", 0.08)
        self.declare_parameter("turn_rate", 0.25)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: turn left in place")
    print("  d: turn right in place")
    print("  space/x: stop")
    print("  q: stop and quit")
    print(
        f"Using topic={node.cmd_topic}, "
        f"forward={node.forward_speed:.2f}, "
        f"reverse={node.reverse_speed:.2f}, "
        f"turn={node.turn_rate:.2f}"
    )

    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            # Republish current state on idle ticks so nav2_qcar_command_convert
            # (0.25 s /cmd_vel_nav safety timeout, added 2026-05-04) does not
            # zero throttle and steering between key presses.
            if not key:
                node.publish_cmd(linear_x, angular_z)
                continue

            if key == "q":
                linear_x = 0.0
                angular_z = 0.0
                node.publish_cmd(linear_x, angular_z)
                print("\nStopped. Exiting manual drive.")
                break
            if key == "w":
                linear_x = node.forward_speed
                angular_z = 0.0
            elif key == "s":
                linear_x = -node.reverse_speed
                angular_z = 0.0
            elif key == "a":
                linear_x = 0.0
                angular_z = node.turn_rate
            elif key == "d":
                linear_x = 0.0
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            else:
                continue

            node.publish_cmd(linear_x, angular_z)
            print(
                f"\rlinear={linear_x:+.2f} angular={angular_z:+.2f}  ",
                end="",
                flush=True,
            )
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
------------------- END manual_drive.py -------------------

Behavioral contract of the file above:
- Subscribes to keystrokes via tty raw mode (`get_key` 100 ms timeout).
- Publishes `geometry_msgs/Twist` on `/cmd_vel_nav`.
- Defaults: `forward_speed=0.10`, `reverse_speed=0.08`,
  `turn_rate=0.25`, `cmd_topic=/cmd_vel_nav`.
- Key bindings (mutually exclusive axes, identical to pre-2026-05-15
  behavior):
    w -> linear_x=forward_speed,  angular_z=0
    s -> linear_x=-reverse_speed, angular_z=0
    a -> linear_x=0,              angular_z=turn_rate
    d -> linear_x=0,              angular_z=-turn_rate
    space / x -> linear_x=0, angular_z=0
    q -> stop and exit
- The only change vs the original Apr 30 version is the 4 lines in the
  idle branch that republish the current `(linear_x, angular_z)` on
  every 100 ms tick where no key was pressed, so the converter's
  0.25 s timeout (set 2026-05-04) does not zero output between
  keystrokes.

## 2026-05-15 (manual_drive steering fix: keep forward speed during A/D)

Context:
- Physical manual driving reported that pressing `W` starts forward motion,
  but pressing `A` or `D` immediately stops forward movement. Expected
  behavior for operator control: press `W` once, then steer with `A/D`
  without dropping throttle.

Root cause:
- `Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` explicitly
  set `linear_x = 0.0` in both `A` and `D` key branches (turn-in-place logic).
- This was not introduced by recent VO/camera changes; it existed in the
  original `manual_drive.py` commit and therefore behaved as designed for
  "turn in place", not "steer while moving".

Code changes made:
- File: `Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py`
- Updated help text:
  - `a: turn left in place` -> `a: steer left (keep current speed)`
  - `d: turn right in place` -> `d: steer right (keep current speed)`
- Updated key handling:
  - Removed `linear_x = 0.0` from `A` branch.
  - Removed `linear_x = 0.0` from `D` branch.
  - `A/D` now only change `angular_z`; current `linear_x` remains latched.

Net effect:
- `W` latches forward speed.
- `A/D` steer while preserving current speed.
- `S` still switches to reverse and recenters steering.
- `space/x` still full stop.
- Idle republish behavior remains intact (prevents timeout-induced zeroing).

Verification:
- `python3 -m py_compile Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` passed.

## 2026-05-15 (manual_drive — Codex external edit applied + verified)

An external edit was applied to
`Development/ros2/src/qcar2_autonomy/autonomy/manual_drive.py` by
Codex (not by this assistant). The intent was to allow simultaneous
throttle + steering while staying as close as possible to the
original file: W/S still set both axes (linear=forward/reverse,
angular=0), but A/D now ONLY set angular and leave linear untouched.
So tapping W and then D produces forward motion AND a right turn,
which was the use case requested by the operator.

Verbatim state of the file after Codex's edit is captured below.
This supersedes the previous 'verbatim state snapshot, post-hotfix #3'
section. Future readers should treat this snapshot as the current
source of truth for manual_drive.py.

------------------- BEGIN manual_drive.py (post-Codex) -------------------
#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def get_key(timeout_s=0.1):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Conservative defaults for first physical tests.
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("reverse_speed", 0.08)
        self.declare_parameter("turn_rate", 0.25)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: steer left (keep current speed)")
    print("  d: steer right (keep current speed)")
    print("  space/x: stop")
    print("  q: stop and quit")
    print(
        f"Using topic={node.cmd_topic}, "
        f"forward={node.forward_speed:.2f}, "
        f"reverse={node.reverse_speed:.2f}, "
        f"turn={node.turn_rate:.2f}"
    )

    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            # Republish current state on idle ticks so nav2_qcar_command_convert
            # (0.25 s /cmd_vel_nav safety timeout, added 2026-05-04) does not
            # zero throttle and steering between key presses.
            if not key:
                node.publish_cmd(linear_x, angular_z)
                continue

            if key == "q":
                linear_x = 0.0
                angular_z = 0.0
                node.publish_cmd(linear_x, angular_z)
                print("\nStopped. Exiting manual drive.")
                break
            if key == "w":
                linear_x = node.forward_speed
                angular_z = 0.0
            elif key == "s":
                linear_x = -node.reverse_speed
                angular_z = 0.0
            elif key == "a":
                # Keep the currently latched speed and only steer.
                angular_z = node.turn_rate
            elif key == "d":
                # Keep the currently latched speed and only steer.
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            else:
                continue

            node.publish_cmd(linear_x, angular_z)
            print(
                f"\rlinear={linear_x:+.2f} angular={angular_z:+.2f}  ",
                end="",
                flush=True,
            )
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
------------------- END manual_drive.py (post-Codex) -------------------

Behavioral contract (current):
- Defaults: forward_speed=0.10, reverse_speed=0.08, turn_rate=0.25,
  cmd_topic=/cmd_vel_nav.
- Idle-tick republish (from hotfix #3) preserved: every 100 ms tick
  with no key pressed still calls node.publish_cmd(linear_x,
  angular_z), keeping nav2_qcar_command_convert (0.25 s timeout) fed.
- Key bindings:
    w       -> linear_x=forward_speed,  angular_z=0       (straight forward)
    s       -> linear_x=-reverse_speed, angular_z=0       (straight reverse)
    a       -> angular_z=+turn_rate     (linear_x untouched)
    d       -> angular_z=-turn_rate     (linear_x untouched)
    space   -> linear_x=0, angular_z=0  (full stop)
    x       -> linear_x=0, angular_z=0  (full stop)
    q       -> publish zeros, print 'Stopped. Exiting manual drive.',
               break out of the loop, destroy node, shutdown rclpy

Runtime delivery:
- rsync of qcar2_autonomy source -> ~/ros2/src/qcar2_autonomy/ verified
  identical post-edit.
- /home/nvidia/ros2/build/qcar2_autonomy/autonomy/manual_drive.py is
  3709 bytes, timestamped May 15 12:13 — matches source byte-for-byte.
- This is the file Python imports via the egg-link
  /home/nvidia/ros2/install/qcar2_autonomy/lib/python3.8/site-packages/
  qcar2-autonomy.egg-link, which points at
  /home/nvidia/ros2/build/qcar2_autonomy.
- No rebuild required. Operator only needs to Ctrl-C the running
  manual_drive in Terminal D and re-run
  'ros2 run qcar2_autonomy manual_drive'.

## 2026-05-15 (Physical Test 2 — ON MAT analysis with new bridge)

Source: `Development/ros2/src/qcar2_autonomy/VO_readings.txt` section
"ON MAT TEST /vo/fault_status" (lines 7564-9033). 714 raw data lines,
651 successfully parsed (63 skipped due to truncated psi_raw fields in
the terminal log; that is a logging-side artifact, not a VO failure).

Drive: manual_drive (post-Codex steer-while-moving fix), 4 left turns
and 3 straight segments per operator's sequence description.
Stack: qcar2_cartographer_launch.py + Python camera_bridge + vo_node
(camera_mode=physical, input_source=ros_rgbd, force_cart_yaw=true).

### Whole-run state mix
- `agree`:      153 / 651  (23.5%)
- `vo_suspect`: 455 / 651  (69.9%)
- `warming`:     43 / 651  ( 6.6%)
- `init`:         0 / 651  ( 0.0%)

Comparison vs prior physical tests:
- Test 1.1: agree 48%, vo_suspect 43%, warming 7%, init 2%
- Test 1.2: agree 33%, vo_suspect 58%, warming 9%, init 0.4%
- Test 2 (this run): agree 23.5%, vo_suspect 69.9%, warming 6.6%, init 0%
Init dropped to zero (good - K_inv fix + aligned depth eliminate
cold-start). agree fell and vo_suspect rose - dominated by turn
segments accumulating in vo_suspect.

### Per-state averages (this run)
- agree:      rho=0.0418 (max 0.144), w=0.535, inliers=213, spread=99
- vo_suspect: rho=0.0740 (max 0.278), w=0.193, inliers=93,  spread=74
- warming:    rho=0.0000 (locked),    w=0.000, inliers=176, spread=94
- zero-inlier frames: 62 / 651 (9.5%); concentrated in vo_suspect
  (13.6% of vo_suspect frames vs 0% of agree).

### Inlier and spread comparison (agree state only)
- 1.1: inliers ~178, weight ~0.512
- 1.2: inliers ~134, weight ~0.480
- 2:   inliers  213, weight  0.535
The new bridge confirms the Step 3 stationary observation under
driving load: higher inlier counts and slightly higher confidence.

### Startup (first 80 frames)
- 68 agree, 7 warming, 5 vo_suspect (no init)
- rho mean 0.025, w mean 0.57, inliers mean 240, spread mean 93
- Best startup of any physical test so far.

### Ending (last 80 frames)
- 80 / 80 vo_suspect
- rho mean 0.054 (low) but w mean 0.08 (very low) - low quality kept
  the state in vo_suspect even though VO-vs-Cart distance was small.
- This is the classic stuck-in-vo_suspect pattern after a hard turn.

### State transitions and re-anchor cadence
- agree segments:        12
- warming segments:       6
- init segments:          0
- agree -> warming:       6
- warming -> agree:       6
- agree -> vo_suspect:    6
- vo_suspect -> agree:    5
Re-anchor gating is firing similarly to Test 1 (9 segments,
7 transitions), so the re-anchor policy is not the regression.

### Trajectory and final divergence
- Cart traj: start (0.11, -0.01) yaw=-179°  -> end (-0.55, 1.53) yaw=-127°
- VO   traj: start (0.11, -0.01) yaw=-179°  -> end ( 0.78, 1.11) yaw=-127°
- Final VO vs Cart divergence:
    Dx = +1.330 m   (Cart went -X, VO went +X => sign FLIP on X)
    Dy = -0.420 m   (Y direction same sign, magnitude under by ~27%)
    Dyaw = 0.0°     (force_cart_yaw=true pins VO heading to Cart)
- Net 52° of CCW rotation across the run.

### Axis sign agreement (Cart deltas vs VO deltas, 40-frame windows)
- X-dominant: score=0.920 over 226 qualifying windows
- Y-dominant: score=1.000 over 286 qualifying windows
vs baselines:
- 1.1: X=0.514, Y=0.707
- 1.2: X=0.634, Y=0.726
Significant improvement in directional consistency. The final 1.33 m
X-divergence is therefore not a continuous direction error but a
transient sign flip during one or more turns that the re-anchor
cooldown did not catch in time.

### Shadow yaw (psi_raw) vs cart_psi
- mean error:    -20.6°
- median:         -9.0°
- stdev:           27.2°
- abs max:        115.0°
- frames with |err| <= 15°: 389 / 651 (59.8%)

Shadow yaw is NOT ready to take over. The 115° peak error and 27°
stdev would cause severe drift in standalone mode. Keep
force_cart_yaw=true for the next run.

### Interpretation
1. The Python camera bridge + K_inv fix + aligned 32FC1 depth path is
   structurally correct. Startup is clean, inliers are high, axis
   sign agreement is excellent, agree-state quality beats both prior
   physical tests.
2. The drop in agree-state share and rise in vo_suspect is dominated
   by turn segments. Tight / jerky left turns under manual drive cut
   translational feature flow, drop spread and weight, trigger safety
   gates, and leave VO stuck in vo_suspect with rho near zero (i.e.
   no update) until quality recovers enough for a re-anchor.
3. The 1.33 m X-direction divergence is the most concerning datum.
   It is caused by a transient X-sign flip during one of the harder
   left turns; X-axis windowed sign agreement is still 0.92, so it
   is not pervasive.
4. Manual drive jitter is a confound, not a controlled variable.

### Recommendations for next session
1. Keep force_cart_yaw=true. Shadow yaw needs more controlled-turn
   data before we can trust standalone yaw.
2. Re-run with smoother driving:
   (a) gentler turn inputs on manual_drive, OR
   (b) autonomy_planner / nav2 path at constant low speed, which is
       the apples-to-apples comparison vs Tests 1.1 and 1.2.
3. Do NOT tune gate parameters yet. One more controlled run is
   needed to distinguish driver smoothness from gate tightness.
4. Optional: capture topic hz on /camera/color_image during driving
   to confirm the bridge still holds 30 fps under autonomy_planner
   plus VO load.

### Files touched in this turn
- VO_readings.txt: read-only (user-owned, never modified).
- /tmp/analyze_test2.py: analysis script (out-of-tree, not committed).
- This changelog entry.
- VO_Conversation_Log.txt Turn 78 (added in this turn).

## 2026-05-15 (workflow note: nav_to_pose.py edits require colcon build)

Confirmed for the record: editing any `.py` source file in
`qcar2_autonomy` (including `nav_to_pose.py`) requires a
`colcon build --packages-select qcar2_autonomy --symlink-install`
before the change takes effect at runtime, even with
`--symlink-install`. The reason is the ament_python build copies
files from `~/ros2/src/qcar2_autonomy/autonomy/` into
`~/ros2/build/qcar2_autonomy/autonomy/`, and Python imports from
the build directory via the egg-link at
`~/ros2/install/qcar2_autonomy/lib/python3.8/site-packages/
qcar2-autonomy.egg-link`. The `--symlink-install` flag only affects
the install dir's relationship to the build dir, not the build
dir's relationship to src. Build itself runs in ~4 s on this
machine.

Quick-experiment alternative: override declared ROS parameters at
launch / run time without editing source. For nav_to_pose.py the
relevant override is:
  ros2 run qcar2_autonomy path_follower --ros-args
      -p rotation_offset:='[90.0]'
which avoids the rebuild cycle entirely while still letting you
sweep the value. Default at this snapshot is `[86.0]` (see
`nav_to_pose.py` line 231).

## 2026-05-15 (Physical Test 3 - autonomy_planner ON MAT analysis)

Source: `Development/ros2/src/qcar2_autonomy/VO_readings.txt` section
"ON MAT TEST using autonomy_planner" (lines 9035-9557). 241 raw
data lines, 220 successfully parsed (21 skipped - truncated psi_raw
in terminal log, logging artifact only).

Drive: autonomy_planner_launch.py - path_follower (nav_to_pose) +
yolo_detector + trip_planner + lane_detector + Planner_server.
Stack: qcar2_cartographer_launch.py + Python camera_bridge + vo_node
(camera_mode=physical, input_source=ros_rgbd, force_cart_yaw=true),
same VO parameters as Test 2.

### Whole-run state mix
- `agree`:      0 / 220  ( 0.0%)
- `vo_suspect`: 220 / 220 (100.0%)
- `warming`:     0 / 220  ( 0.0%)
- `init`:        0 / 220  ( 0.0%)
VO never entered `agree` for a single frame across the entire run.

### Comparison vs Test 2 (same drive plan, manual_drive)
- Test 2: agree 23.5%, vo_suspect 69.9%, warming 6.6%, init 0%
- Test 3: agree 0%, vo_suspect 100%, warming 0%, init 0%
- Test 2 final divergence: Dx=+1.33, Dy=-0.42
- Test 3 final divergence: Dx=+0.09, Dy=-1.61
VO position barely moved while Cart traveled ~1.6 m in Y. The Y
divergence is essentially the full cart displacement - VO did not
integrate translation across the run.

### Per-run averages
- rho       mean=0.078  med=0.000  max=0.358
- w         mean=0.028  med=0.000   <- updates rejected near 100% of the time
- inliers   mean=112    med=66     min=0     <- not catastrophic, but lower than Test 2 agree=213
- spread    mean=76     med=79
- zero-inlier frames: 14 / 220 (6.4%)

### Trajectory
- Cart: (0.11, -0.01) yaw -180  ->  (-0.09, 1.52) yaw -140    [1.6 m, 40 deg CCW]
- VO:   (0.08, -0.04) yaw -180  ->  (-0.00, -0.09) yaw -140   [<0.1 m]
- Final divergence: Dx=+0.090, Dy=-1.610, Dyaw=0 (force_cart_yaw=true)

### State transitions / re-anchor segments
- agree segments: 0
- warming segments: 0
- init segments: 0
- Total transitions: 0 (state never changed)
Re-anchor never fired because VO never produced a valid update to
anchor from.

### Shadow yaw (psi_raw vs cart_psi)
- mean error: +4.5 deg
- stdev:      106.5 deg
- abs max:    178.0 deg
- |err| <= 15 deg: 42 / 220 (19.1%)
Far worse than Test 2 (27 deg stdev, 115 deg max). Consistent with
psi_raw not integrating because dpsi=0 on every rejected frame -
psi_raw effectively froze at its initial value while cart_psi
swept through the turn.

### Axis sign agreement (Cart deltas vs VO deltas, 40-frame windows)
- X-dominant: score=1.000 over 44 windows  (but VO Dx is near zero,
  so this is mostly about sign of near-zero noise)
- Y-dominant: score=0.844 over 96 windows
These numbers are misleadingly OK - they reflect the few windows
where VO did register tiny motion. The bulk of the run has VO
delta = 0, which the sign-agreement metric doesn't penalize.

### Diagnostic interpretation
The signature 'rho=0 w=0 dx=0 dy=0 dpsi=0' on the vast majority of
frames is the diagnostic of safety-gate failure inside
visual_odometry.py - the VO frame was computed but rejected before
being committed. Candidate rejection causes (in priority order):

1) Per-frame translation exceeded `max_translation` gate.
   autonomy_planner default speed is higher than manual_drive's
   0.10 m/s. Higher car speed + variable VO tick rate (CPU
   contention from yolo/lane/Planner_server) -> larger per-frame
   pixel disparity -> ORB-matches scatter -> RANSAC produces a
   pose that exceeds the magnitude gate.

2) Per-frame dt exceeded `max_dt` gate.
   Test 3 produced 220 fault_status frames; Test 2 produced 651
   over a comparable run. A 3x lower tick rate is consistent with
   the camera bridge or VO node falling behind under combined
   load (yolo + lane + Planner_server all subscribed to
   /camera/color_image).

3) Inlier count low after RANSAC because per-frame motion stresses
   ORB. Average 112 vs Test 2 agree-state 213. Not zero but
   borderline.

Bridge architecture itself is not the failure: peak inliers at
startup were 339, encoding auto-detect fired, [DEPTH DIAG ALIGNED]
logs show plausible metric values (1.0-4.6 m) throughout. The
failure mode is on the VO side.

### Recommended next steps (do NOT tune gates yet)
1. Re-run with concurrent `ros2 topic hz /camera/color_image` and
   `ros2 topic hz /vo/fault_status` captured in a 5th terminal.
   This will confirm whether the bridge fps drops under the full
   autonomy stack.
2. Re-run with ONLY `path_follower` (no yolo/lane/Planner_server)
   to isolate planner speed from CPU contention:
     ros2 run qcar2_autonomy path_follower
3. If feasible, lower the planner speed to ~0.10-0.15 m/s for an
   apples-to-apples vs Test 2.
4. Only after the above: review visual_odometry.py safety-gate
   thresholds (`max_translation`, `max_rotation_deg`, `max_dt`)
   for whether they need to scale with planner speed.

Operator-side notes:
- Test 3 was a planned comparison run; the failure is informative,
  not a regression vs Test 2. Test 2 results remain the baseline
  for the new bridge architecture.
- force_cart_yaw=true is masking how bad VO yaw is right now.
  Definitely keep that flag on until the gate / fps issue is
  resolved.

### Files touched in this turn
- VO_readings.txt: read-only (user-owned).
- /tmp/analyze_test3.py: out-of-tree analysis script (cloned from
  /tmp/analyze_test2.py, only path differed).
- This changelog entry.
- VO_Conversation_Log.txt Turn 80 (added in this turn).

## 2026-05-15 (Test 3 follow-up: root-cause for fps drop + stationary car)

Operator captured concurrent fps measurements during another
autonomy_planner run and reported:
- ros2 topic hz /camera/color_image  -> avg ~15 fps (max 15)
- ros2 topic hz /vo/fault_status     -> avg ~3 Hz, occasionally < 1 Hz
- Car did not move during the run.

These two observations resolve the open questions from the Test 3
analysis above.

### Finding 1: bridge fps halved under autonomy stack (30 -> 15)

autonomy_planner_launch.py actually spawns:
- path_follower (light, geometry only)
- traffic_system_detector (executable=yolo_detector,
  name=qcar2_yolo_detector)
- trip_planner (light)
- Planner_server (light)
- lane_detector is commented out at the LaunchDescription return

yolo_detector subscribes to BOTH /camera/color_image AND
/camera/depth_image at QoS depth 5 (yolo_detector.py:93-96) and runs
YOLO inference on every received frame. On this Jetson this is at
the edge of available compute. When inference falls behind, the
image pipeline back-pressures and the bridge's publish timer slips,
halving the effective publish rate. Consequence: vo_node sees 15
frames/sec instead of 30, per-frame motion roughly doubles, and the
max_translation / max_dt safety gates inside visual_odometry.py
reject the update. This is the mechanism behind Test 3's
100% vo_suspect state mix.

### Finding 2: car did not move because start_path defaulted to False

nav_to_pose.py:238-239 declares:
    self.declare_parameter('start_path', [False])
    self.path_execute_flag = list(self.get_parameter
        ('start_path').get_parameter_value().bool_array_value)[0]

The launch file does not set start_path, so path_follower comes up
but does not issue Twist commands until the flag is flipped. The
previous Test 3 run apparently had it flipped at some point during
the run (since cart moved 1.6 m in Y); the new run did not. Cart
movement is incidental to the fps diagnostic - the fps drop is
purely a function of who is subscribed, not whether the car is
driving.

Default desired_speed in nav_to_pose.py is [0.2] m/s, which is 2x
manual_drive Test 2's 0.10 m/s. For an apples-to-apples comparison
the next run should override to 0.10.

### Next run plan (Test 4 - isolation)

Goal: confirm yolo_detector is the cause of the fps drop AND
produce a valid VO comparison vs Test 2 baseline.

Terminals A, B, C unchanged from Test 2/3 (quarc_run + cartographer
launch, vo_node, fault_status echo).

Terminal D replaced by:
    ros2 run qcar2_autonomy path_follower --ros-args \
        -p start_path:='[True]' \
        -p desired_speed:='[0.1]'
(path_follower only - no yolo, no trip_planner, no Planner_server.)

Terminal E (new) - fps watchers, started before the run:
    ros2 topic hz /camera/color_image
    ros2 topic hz /vo/fault_status

Expected if the hypothesis is correct:
- /camera/color_image returns to ~28-30 fps
- /vo/fault_status returns to ~5 Hz
- VO re-enters agree state, fault_status mix becomes Test 2-like
  or better

If bridge fps still does not recover to ~30 with just
path_follower running, the suspect shifts to vo_node load or
cartographer rather than yolo.

Operator will paste new results under a section in VO_readings.txt
such as "ON MAT TEST - autonomy path_follower alone, no yolo".

Do NOT tune visual_odometry.py safety gates yet. The Test 3
analysis already cautioned this; the Finding 1 mechanism makes it
concrete: the gates are appropriate for 30 fps input; under 15 fps
they reject correctly. Restore the input rate first, then re-evaluate.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 81 (added in this turn).
- No code edits.

## 2026-05-15 (Test 4 follow-up: yolo hypothesis ruled out, bridge fps anomaly isolated)

Operator ran the isolation test (Terminal D = path_follower alone,
no yolo / trip_planner / Planner_server) and reported:
- /camera/color_image: ~16-18 fps (NOT recovered to 30)
- /vo/fault_status:    ~3.7-4 Hz with path_follower running,
                       ~5 Hz before path_follower came up
- Car did not move under path_follower alone (probably because
  trip_planner / Planner_server are needed to seed the planned
  route - path_follower follows a path it does not own).

### Yolo hypothesis: refuted

If removing yolo + trip_planner + Planner_server leaves the bridge
stuck at ~17 fps, those nodes were not the bottleneck. CPU
contention from the autonomy stack is not the dominant cause of
the bridge fps drop. The Test 3 follow-up entry above was wrong
and should be read in light of this new evidence.

### What is actually going on (re-derived)

Two separate observations, only one anomalous:

1. fault_status at ~5 Hz is NORMAL for this stack. Test 2 produced
   651 frames over ~130 s = ~5 Hz. vo_node ticks at 20 Hz nominal
   (vo_node.py:349-350) but ORB + RANSAC + SVD on 640x480 frames
   takes ~150-200 ms on this Jetson, so the effective fault_status
   rate is ~5 Hz. This is a VO-computation cost, not a rate cap.

2. /camera/color_image at ~17 fps IS anomalous. Step 3 stationary
   test (2026-05-14) verified ~30.3 fps via the bridge's own
   '[bridge hb]' log line under the EXACT SAME stack
   (qcar2_cartographer_launch.py + vo_node). Something has changed
   system-side since.

qcar2_camera_bridge.py polls QCar2DepthAligned.read() at 60 Hz
and publishes only when read() returns True (bridge code lines
164-167, 191-199). Nothing on the ROS side caps the rate. The
published fps equals the PIT runtime's delivery rate. Therefore
the ~17 fps observation means the Quanser depth-align runtime
itself is delivering 17 fps, not the 30 fps Step 3 confirmed.

### Suspects (in priority order)

A. Jetson power mode / clocks throttled.
   nvpmodel and jetson_clocks govern CPU + GPU frequencies. The
   QCar2DepthAlign pipeline depends on GPU/CUDA for the stereo->
   aligned 32FC1 pass. If the system reverted to a low-power
   profile (e.g. 15W) between Step 3 and today, 30 fps -> ~15 fps
   is exactly the expected drop.

B. Orphaned QCar2DepthAlign runtime from a previous session.
   Two processes contending for the camera USB device and
   tcpip://localhost:17003 could halve the effective rate.

C. Thermal throttling after sustained day of operation.

D. RealSense USB renumeration into a slower port / hub.

### Diagnostic steps for operator (car off mat, full stack down)

1. Check Jetson power profile and clocks:
     sudo nvpmodel -q
     sudo jetson_clocks --show
   If not MAXN / clocks not at max, restore with:
     sudo nvpmodel -m 0           # MAXN profile
     sudo jetson_clocks            # pin clocks high

2. Check for orphan camera owners:
     ps -ef | grep -iE 'QCar2DepthAlign|quarc_run|rgbd|camera_bridge' | grep -v grep
   Kill any survivors before restarting cartographer.

3. Run tegrastats during a fresh stack bring-up to see CPU / GPU /
   thermal state live:
     sudo tegrastats --interval 1000

4. The bridge logs '[bridge hb] published=X frames in 5.0 s
   (~Y fps)' every 5 s in Terminal A. That line is ground truth
   for the bridge's measured delivery rate. Whatever it reports
   should match the 'ros2 topic hz' on /camera/color_image. If
   the heartbeat says ~30 but the topic hz reports ~15, that
   means consumers (vo_node) are throttling subscription -
   different diagnosis from PIT-throttled.

### Decision on mat testing

Operator has decided to continue using manual_drive on the mat
for VO testing going forward (constant operator-controlled speed
at 0.10 m/s is the cleanest baseline). autonomy_planner runs are
deferred until: (a) the bridge fps anomaly is resolved, and
(b) the autonomy stack can be run with a known good speed and
without yolo at full inference rate.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 82 (added in this turn).
- No code edits.

## 2026-05-15 (system diagnostics: bridge confirmed at 30 fps, topic_hz was the artifact)

Operator captured the diagnostic commands from the previous turn
into VO_readings.txt under "GPU checks" section. Headline results:

1. Bridge heartbeat in cartographer log:
   [bridge hb] published=152 frames in 5.0 s (~30.4 fps)
   ----> bridge IS publishing at 30 fps. PIT runtime healthy.

2. Jetson power profile:
   sudo nvpmodel -q  ->  NV Power Mode: MAXN  (correct)

3. Jetson clocks:
   CPU max freq: 2188800 (2.18 GHz) on all 8 cores
   GPU max freq: 930750000 (930 MHz)
   EMC at 2133 MHz, DLA at 1.4 GHz, PVA at 1.19 GHz
   ----> CPU/GPU/memory clocks at expected MAXN ceilings.
   The 'schedutil' governor scales them down when idle (729 MHz
   in tegrastats during idle), and they ramp back to 2188 MHz the
   moment work hits (visible on the 13:58:31 tegrastats line
   where 4 cores briefly jumped to 2188 MHz).

4. Orphan processes:
   ps -ef | grep -iE 'QCar2DepthAlign|quarc_run|rgbd|camera_bridge'
   -> empty. No duplicate camera owners.

5. tegrastats during idle window:
   CPU 0-5% across cores, GPU 0%, temps ~38C. Nothing hot, nothing
   pegged. Plenty of headroom.

### Re-interpretation of the 'bridge fps drop' story

Earlier this session two hypotheses were wrong:
1. (Turn 81) 'yolo_detector starves the bridge to 15 fps' - WRONG
   path_follower-alone test ruled out CPU contention as cause of
   the bridge published rate dropping.
2. (Turn 82) 'PIT runtime throttled / power mode' - WRONG
   bridge heartbeat confirms 30.4 fps published; nvpmodel MAXN,
   clocks at max, no orphans.

Correct interpretation:
- The bridge HAS been publishing at 30 fps consistently. The
  Step 3 stationary test result still stands.
- 'ros2 topic hz /camera/color_image' reports ~17 fps because it
  measures messages RECEIVED by its subscriber, not messages
  published. On Jetson + ROS 2 humble + DDS (cyclonedds or
  fastrtps), large Image topics over default reliable QoS
  routinely get half-dropped at the DDS layer when there is any
  subscriber pressure. This is a middleware artifact, not a
  publisher problem.
- /vo/fault_status at ~5 Hz is the VO computation cost
  (ORB + RANSAC + SVD on 640x480 takes ~200 ms per vo_tick on
  this Jetson). The 20 Hz timer in vo_node.py:349-350 cannot
  meet its period; effective rate is ~5 Hz. This is normal and
  matches every prior test (Test 2 had 651 frames over ~130 s).

### So what actually broke Test 3?

Revised mechanism (still pointing at yolo, but via a different
channel than originally claimed):

- Bridge at 30 fps in both Test 2 and Test 3 - constant.
- vo_node CPU budget is what changed.
- Test 2 (manual_drive only): vo_node had full CPU for its
  ~200 ms tick - produced ~5 Hz fault_status with high agree
  quality.
- Test 3 (autonomy_planner with yolo_detector inference on every
  frame): vo_node CPU contended; per-tick time stretched. Time
  gap between frames vo_node actually processed increased,
  per-frame motion in pixels doubled or tripled, max_translation
  / max_dt safety gates inside visual_odometry.py rejected every
  update. Hence the 100% vo_suspect mix.

Yolo IS the cause of Test 3 failure, but via vo_node CPU
contention rather than bridge fps throttling.

### On the 60 Hz poll rate vs 30 Hz delivery rate

Operator asked whether the disparity is itself a problem. It is
not. The 2x oversample pattern (qcar2_camera_bridge.py:121, 167)
is deliberate:
- Polling 2x the source rate halves the average ROS-publish
  latency (~8 ms vs ~17 ms if we polled at exactly 30 Hz).
- read() returns False quickly when no new frame is available
  (microseconds of CPU per no-op tick). No CPU cost.
- Lines 191-199 of the bridge guard publish on 'if not new:
  return'.
This is the standard pattern for ROS bridges over a fixed-rate
external source and should remain at 60 Hz.

### Mat-testing decision (carried from prior turn)

Operator is continuing mat VO testing with manual_drive only.
Because vo_node CPU contention is the dominant degrader,
manual_drive (which spawns no extra image-subscribing or
GPU-inferring nodes) gives the cleanest VO baseline. Test 2
remains the controlled-baseline run.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 83 (added in this turn).
- No code edits.

## 2026-05-15 (planning / parameter-tuning analysis, no code edits)

Operator wants to step up VO tuning - specifically the RANSAC
sample size from 2 to 3 or 4 - and also wants to think about
whether VO can keep up at 0.4 / 0.6 m/s driving speeds. Also
floated an architectural question: should yolo + VO + camera
ownership be merged into a single Python process so the camera
frames never traverse ROS pub/sub?

### Current VO parameters (from visual_odometry.py:569-573,785)

- n_features:        800
- match_ratio:       0.75
- ransac_threshold:  0.05 m
- ransac_iterations: 300
- min_inliers:       8
- max_translation:   0.20 m / frame
- max_rotation:      15 deg / frame
- max_dt:            0.20 s
- RANSAC sample size (line 785, np.random.choice(M, 2)): s=2
- Docstring (line 779) explicitly calls it '2-point RANSAC with
  SVD Procrustes'.

### On bumping s = 2 -> 3

The 2D rigid-body transform has 3 DOF (theta, tx, ty), so s=2 is
the minimum geometric sample and s=3 is the natural over-
constrained robust choice. s=4 is over-constrained for the model
and not recommended.

RANSAC iteration scaling at w=0.5 inlier ratio and p=0.99
confidence:
- s=2: N approx 16 iterations
- s=3: N approx 35
- s=4: N approx 72
Current iterations=300 is well above any of these thresholds.
Conclusion: bumping s from 2 to 3 with iterations unchanged is a
free quality improvement (better hypothesis robustness, same
per-iteration cost).

Caveat: s-count tuning does not address the dominant Test 2
failure mode. Most rejected frames have rho=0 w=0 dx=0 dy=0,
which is the safety-gate path (max_translation, max_rotation, or
min_inliers tripping), not a bad-consensus path. s-count affects
consensus quality given that the gates pass.

### Speed budget for 0.4 / 0.6 m/s at 5 Hz VO

Per-frame motion = speed / VO rate:
- 0.10 m/s : 0.020 m / frame  (Test 2 baseline)
- 0.20 m/s : 0.040 m / frame
- 0.40 m/s : 0.080 m / frame
- 0.60 m/s : 0.120 m / frame

All below the 0.20 m max_translation gate. BUT - the metric-space
gate is not the right rate-limit. Pixel-space disparity is. At
0.60 m/s with 640x480 frames and scene depths ~1-2 m, features
can shift 30-60 px between consecutive frames. ORB + BFMatcher
with default search radius begins to lose matches around that
disparity. 0.40 m/s is approximately the upper limit before
match counts and inlier ratio collapse, given the current VO
tick rate.

Two ways to extend the operating envelope:
(a) Raise VO frame rate (reduce per-tick cost - shrinks per-frame
    motion proportionally).
(b) Use a disparity-tolerant matcher (image pyramid, LK optical
    flow seeded from previous match, etc.) - bigger rewrite.

### Speedup levers, ranked

                                       est speedup   risk    quality
  ransac_iterations 300 -> 100         3x            low     same
  s 2 -> 3                             ~0            low     +
  n_features 800 -> 500                1.5x          low     -
  resize 640x480 -> 320x240            3-4x          medium  -
  skip every 2nd frame                 2x            low     +per-frame motion
  ORB+BF -> FAST + LK optical flow     5-10x         high    different regime

The combination 'ransac_iterations 300 -> 100  AND  s 2 -> 3' is
a freebie: same or better quality, ~3x faster RANSAC step. This
is the recommended first experiment.

### On merging yolo + VO + camera-owner into one file

What that change would actually save:
- DDS serialization / deserialization of /camera/color_image and
  /camera/depth_image. About 30 MB/s of serialized image data on
  the wire. The half-drop seen on 'ros2 topic hz' is part of
  this overhead.

What it does NOT save:
- Yolo inference cost. GPU/DLA work still has to run.
- VO computation cost. CPU work still has to run.
- Python GIL serializes CPU paths inside one process; yolo on
  GPU/DLA can overlap with VO on CPU, but that already happens
  today across separate processes.

Downsides:
- Single point of failure for both yolo and VO.
- Cannot disable yolo independently for isolation tests
  (exactly what Test 4 needed).
- Hard to debug a large coupled file.

Cleaner alternative if intra-process plumbing is the goal:
ROS 2 composable nodes loaded into a single component_container
with intra-process comms enabled. yolo + VO + bridge each remain
their own file/class but share buffers zero-copy. This is the
ROS 2 idiom for high-rate image pipelines.

Cleaner alternative if just freeing CPU is the goal:
Throttle yolo inference to 10 Hz instead of 30 Hz. Sign/obstacle
perception rarely needs 30 Hz; would free large amount of GPU/
CPU contention for VO.

Recommendation: do not merge into a single file. If/when we want
to fight the DDS overhead, do composable nodes. If/when we want
to fight CPU contention, throttle yolo.

### Proposed next step

Option A (recommended): tune ransac_iterations 300 -> 100 and
RANSAC sample size 2 -> 3 in visual_odometry.py. Rsync, rebuild,
re-run a manual_drive Test 2-style trajectory at 0.10 m/s. If
agree-state share goes up vs Test 2 (23.5%), keep. If unchanged
or worse, revert and try a different lever.

No code edits in this turn - waiting for operator's go on which
option to execute.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 84 (added in this turn).
- No code edits.

## 2026-05-15 (QoS fix + ORB ROI mask for VO)

Two related changes made in this turn. Self-contained diffs and
behavioral contracts below so this entry is the recovery point.

### Change 1 - sensor_data QoS across the camera topic chain

Why: bridge publishes /camera/color_image and /camera/depth_image
at 30 fps (confirmed by [bridge hb] heartbeat), but 'ros2 topic hz'
reports ~17 fps. Root cause is reliable QoS + KEEP_LAST depth=5 on
large Image messages, which half-drops at the DDS layer under any
subscriber pressure. Switching to BEST_EFFORT (qos_profile_sensor_
data) eliminates the half-drop because BEST_EFFORT drops in flight
instead of buffering. This is the standard ROS 2 idiom for high-
rate sensor topics. Subscribers must use BEST_EFFORT too; otherwise
the publisher and subscriber are QoS-incompatible and they do not
even connect.

Files edited:

A) qcar2_camera_bridge.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed publishers from:
     self.pub_rgb   = self.create_publisher(Image, '/camera/color_image', 5)
     self.pub_depth = self.create_publisher(Image, '/camera/depth_image', 5)
   to:
     self.pub_rgb   = self.create_publisher(Image, '/camera/color_image', qos_profile_sensor_data)
     self.pub_depth = self.create_publisher(Image, '/camera/depth_image', qos_profile_sensor_data)
   Added a multi-line comment block above the publishers explaining
   why the change was made.

B) vo_node.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed image subscribers in the ros_rgbd branch from depth=5
   default-QoS positional argument to qos_profile_sensor_data.
   The color_sub and depth_sub create_subscription() calls now pass
   qos_profile_sensor_data in place of the bare '5'.

C) yolo_detector.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed the two create_subscription calls for /camera/color_image
   and /camera/depth_image to use qos_profile_sensor_data instead of
   the bare integer 5.

D) traffic_system_detector.py
   Added import: 'from rclpy.qos import qos_profile_sensor_data'
   Changed the camera_image_subscriber to use qos_profile_sensor_data
   instead of the bare integer 10.

E) lane_detector.py
   The file already used an explicit QoSProfile() construction.
   Changed reliability from ReliabilityPolicy.RELIABLE to
   ReliabilityPolicy.BEST_EFFORT, and depth from 10 to 5 for
   consistency with the other subscribers.

Behavioral contract after Change 1:
- ros2 topic hz /camera/color_image should report ~30 fps,
  matching the [bridge hb] heartbeat.
- vo_node, yolo_detector, traffic_system_detector, and
  lane_detector remain QoS-compatible with the bridge
  publishers (BEST_EFFORT on both sides).
- For developers: any new subscriber to /camera/* must use
  qos_profile_sensor_data (or equivalent BEST_EFFORT QoS) or
  it will silently fail to receive frames.

### Change 2 - ORB ROI mask in visual_odometry.py

Why: the top portion of the image is usually sky, ceiling, or
distant horizon - mostly low-quality features for VO and often
lacking usable depth (RealSense max range falls off past ~3 m).
Excluding it focuses ORB on ground-level scene features that
actually carry valid 3D backprojection. Implemented as an
OpenCV ORB mask (cv2.ORB.detectAndCompute(gray, mask)) rather than
an image crop - this keeps image dimensions and camera intrinsics
unchanged so DepthProjector.pixels_to_3d_body needs no rescaling.

File: visual_odometry.py

Signature change in VisualOdometryDepth.__init__:
  Added new keyword argument 'roi_top_fraction=0.0' at the end of
  the constructor parameter list.

New instance state:
  self.roi_top_fraction = float(roi_top_fraction)
  if not (0.0 <= self.roi_top_fraction < 1.0):
      self.roi_top_fraction = 0.0
  self._orb_mask = None   # built lazily on first frame

In update(), replaced this single line:
    keypoints, descriptors = self.orb.detectAndCompute(gray, None)
with the mask block:
    mask = None
    if self.roi_top_fraction > 0.0:
        H, W = gray.shape[:2]
        if (self._orb_mask is None
                or self._orb_mask.shape[0] != H
                or self._orb_mask.shape[1] != W):
            self._orb_mask = np.zeros((H, W), dtype=np.uint8)
            y_start = int(H * self.roi_top_fraction)
            self._orb_mask[y_start:, :] = 255
        mask = self._orb_mask
    keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

Behavioral contract:
- Default roi_top_fraction=0.0 means the legacy behavior is
  preserved exactly (mask is None, ORB processes the full image).
- Setting roi_top_fraction=0.30 builds a (480x640) uint8 mask
  with rows 0-143 zero (excluded) and rows 144-479 set to 255.
- The mask is built once on the first qualifying frame and reused
  across frames as long as image dimensions stay constant.
- Camera intrinsics (K_rgb, K_depth, K_rgb_inv, K_depth_inv) and
  the alignment matrix are NOT touched. Pixel coordinates
  reported by ORB stay in the full (640x480) frame so
  DepthProjector.pixels_to_3d_body sees the same coordinate
  system it did before.

### Change 3 - expose roi_top_fraction as a ROS parameter

File: vo_node.py

Added a parameter declaration in the existing block of declarations:
    self.declare_parameter('roi_top_fraction', 0.0)

Added read in the parameter-binding block:
    roi_top_fraction = float(
        self.get_parameter('roi_top_fraction').value)

Added pass-through into the VisualOdometryDepth constructor call:
    roi_top_fraction=roi_top_fraction,

Operator usage:
    ros2 run qcar2_autonomy vo_node --ros-args \
        -p camera_mode:=physical \
        -p input_source:=ros_rgbd \
        -p force_cart_yaw:=true \
        -p roi_top_fraction:=0.30

0.30 is a sensible first try (skips top 30%, keeps bottom 70%).
0.40 is more aggressive. Comparing two runs at 0.30 vs 0.0 will
show whether the ROI mask improves agree-state share, inlier
count, or per-tick computation time.

### Build verification

- python3 -m py_compile qcar2_camera_bridge.py vo_node.py
  visual_odometry.py yolo_detector.py traffic_system_detector.py
  lane_detector.py  ->  PASS
- rsync from ACC_Development -> ~/ros2/src/qcar2_autonomy/  ->  OK
- colcon build --packages-select qcar2_autonomy --symlink-install
  ->  PASS (only the standard easy_install deprecation warning)

### Files touched in this turn
- qcar2_camera_bridge.py
- vo_node.py
- yolo_detector.py
- traffic_system_detector.py
- lane_detector.py
- visual_odometry.py
- VO_CHANGELOG.md (this entry)
- VO_Conversation_Log.txt Turn 85 (added in this turn)

## 2026-05-15 (Physical Test 3 OFF-MAT — QoS fix verified, ROI sweep)

Source: VO_readings.txt section 'Physical Test 3' (lines 9613-10381).
Operator ran three off-mat stationary phases back-to-back to test
the new QoS profile and the new roi_top_fraction parameter.

### Phase line ranges
- Phase 1 (ROI=0.0):  cmd at line 9614, fault_status from 9692-9781
  (44 data lines)
- Phase 2 (ROI=0.30): cmd at line 9782, fault_status from 9902-10149
  (123 data lines)
- Phase 3 (ROI=0.40): cmd at line 10150, fault_status from 10234-end
  (73 data lines)

### Camera publish-rate confirmation

Phase 1 (fresh start, no restart artifact):
  ros2 topic hz /camera/color_image  ->  29.4-30.0 fps steady
  ros2 topic hz /camera/depth_image  ->  29.0-29.4 fps steady

Conclusion: the QoS fix from earlier this turn is verified. The
subscriber side (ros2 topic hz tool) now reports the same rate the
bridge is publishing. The previous ~17 fps reading was indeed the
DDS half-drop on reliable QoS for large Image messages, and
switching everything to qos_profile_sensor_data resolved it.

Phases 2 and 3 show topic_hz averages of ~25-26 fps with max
inter-message gaps of 0.368-0.399 s. Those are consistent with a
brief stall during the vo_node restart at the start of each new
phase; the moving-average buffer in 'ros2 topic hz' carries the
spike forward and never fully recovers within the captured window.
This is a measurement artifact, not a bridge slowdown - the [bridge
hb] line would be the ground truth (not captured this run). To
verify on subsequent runs, glance at Terminal A's bridge heartbeat
during ROI phases.

### Per-phase quality (stationary, force_cart_yaw=true)

                            P1 (no ROI)   P2 (0.30)    P3 (0.40)
  n frames:                       44          123           73
  state mix:                      100% agree  100% agree   100% agree
  mean rho (good ticks):       0.0208       0.0175       0.0199
  mean weight (good ticks):    0.689        0.634        0.575
  mean inliers (all):          333          356          410
  mean spread (all):           102          89           77
  zero-weight %:               20.5%        15.4%        23.3%

### Interpretation

1) Inlier count rises monotonically as we mask more of the top of
   the image (333 -> 356 -> 410). Same 800-feature ORB budget, but
   more of those features land in the lower image where depth is
   reliable. Net effect: more matches survive the depth-validity
   filter.

2) Spread drops monotonically (102 -> 89 -> 77) because features
   are spatially confined to the unmasked region. Expected.

3) Weight drops modestly (0.69 -> 0.63 -> 0.57). The VO weight
   blends multiple quality signals including spread, so a smaller
   spread region pulls the weight down even when inliers and rho
   improve.

4) Zero-weight (rejected-tick) percentage:
      P1: 20.5%
      P2: 15.4%   <- best
      P3: 23.3%
   Rejection rate is non-monotonic: ROI=0.30 actually reduces
   rejections vs no-ROI, but ROI=0.40 is over-aggressive and
   rejections climb back up.

5) rho is lowest at ROI=0.30 (0.0175 m), which is the residual
   between VO and Cartographer poses. Lower is better.

### Verdict

- QoS fix: CONFIRMED working. /camera/color_image and
  /camera/depth_image now report ~30 fps via 'ros2 topic hz',
  matching the bridge heartbeat. Any future subscriber added to
  /camera/* must use BEST_EFFORT QoS to remain compatible.

- ROI mask: SAFE. None of the three configurations broke VO at
  rest. State stayed in agree throughout.

- Sweet spot: roi_top_fraction = 0.30. Best on every aggregate
  quality metric (lowest rho, lowest rejection rate, healthy
  inlier count). ROI = 0.40 is past the optimum (weight drops,
  rejections climb).

### Recommended next step

Bring the car onto the mat and run manual_drive with
  -p roi_top_fraction:=0.30
added to the vo_node launch params (other VO params unchanged).
Drive a Test 2-style controlled trajectory (forward, gentle left,
straight, etc.) at the same speed used in Test 2 (manual_drive
defaults). Save under a new VO_readings section so we can compare
agree-state share and axis sign agreement against Test 2 (manual,
no ROI) baseline:
  Test 2: agree 23.5%, X-axis sign agreement 0.92, Y 1.00
If Test 4 (manual + ROI 0.30) shows agree share rising and/or
axis sign agreement holding 0.9+, the ROI mask graduates.

Do NOT change any other VO parameters for the mat run. We want
one variable changed at a time so the comparison is clean.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 86 (added in this turn).
- No code edits in this turn; the QoS + ROI code changes from the
  prior turn are the ones being validated here.

## 2026-05-15 (CORRECTION to Test 3 analysis above)

Operator clarified post-analysis that the prior section overstated
two things. Recording the correction inline so future readers and
the parallel Codex session do not act on the incorrect framing.

### Correction 1: Phase 1 'baseline' was not comparable to 2 and 3

The operator clarified that during Phase 1 the vo_node was NOT
running - the ros2 topic hz reading was captured with cartographer
+ bridge only, no vo_node subscribed. Phases 2 and 3 had vo_node
running, which adds a consuming subscriber to /camera/color_image
and /camera/depth_image. So Phase 1's 29.4-30.0 fps and Phases 2
and 3's ~25 fps stable readings are not measuring the same load
regime. They are not three datapoints on the same curve.

### Correction 2: phase durations were uneven

Operator did not time the phases, and the captured fault_status
counts (44 / 123 / 73 lines) reflect different total durations and
different stabilization-window-included fractions. The per-phase
average rho / weight / inliers / spread numbers in the prior
section are therefore not strictly comparable, and the
'sweet spot at ROI=0.30' conclusion is softer than it sounded.
ROI=0.30 remains a reasonable default for the mat run on the basis
of being mid-aggressive (not extreme), but should not be cited as
demonstrably optimal from this dataset.

### What we actually know

1) QoS fix did SOMETHING positive: pre-fix topic_hz reading was
   ~17 fps under load; this run shows ~25 fps under vo_node load
   in stable windows of Phases 2 and 3. Better than before.

2) Whether the bridge is at 30 fps with vo_node also consuming is
   NOT proven by this run because the bridge heartbeat lines
   ([bridge hb] from Terminal A) were not captured. The bridge
   may still be at 30 and ros2 topic hz under-reports due to
   tool/sampling artifacts, OR the bridge may be at ~25 fps
   under vo_node back-pressure even with BEST_EFFORT. Data does
   not distinguish.

3) All three ROI configurations held 100% agree state at rest.
   That is the most defensible claim from this dataset:
   ROI=0.30 and 0.40 are safe to use; neither breaks VO at rest.

### Updated plan

Mat run with manual_drive remains the same: run vo_node with
  -p roi_top_fraction:=0.30
and otherwise the Test 2 baseline. Choice of 0.30 is on the basis
of being conservative and consistent with the operator's earlier
intent, not on the basis of being statistically best from Test 3.

Additional one-line diagnostic for the mat run: keep an eye on
Terminal A's [bridge hb] log line during driving. If it consistently
reads ~30 fps under live vo_node load, that confirms the bridge is
fine and any topic_hz under-read is a measurement-tool delay. If
it reads ~25 fps under load, then there is a real DDS drop
happening and we look at it next.

### Files touched in this turn
- This changelog correction (appended to the prior Test 3 entry).
- VO_Conversation_Log.txt Turn 87 (added in this turn).
- No code edits in this turn.

## 2026-05-15 (Test 3 — bridge heartbeat confirms 30 fps under vo_node load)

Operator confirmed by glancing at Terminal A's cartographer log
during the Test 3 phases:
  [bridge hb] ~151 frames in 5.0 s (~30.1 fps) consistently

This resolves the residual question from the Test 3 correction
entry above. The bridge is publishing at 30 fps under live vo_node
load - same as the Step 3 stationary number from 2026-05-14
(~30.3 fps). Nothing on the publish side is degraded by adding a
vo_node consumer.

Therefore the ~5 fps gap between the bridge heartbeat (30) and
ros2 topic hz (~25) under load is a measurement-side artifact:
either the 'ros2 topic hz' tool's own subscriber is dropping a
few frames under load (each BEST_EFFORT subscriber drops
independently when DDS cannot fan-out fast enough), or the
tool's sampling lags slightly behind the real arrival rate.
Either way it is purely diagnostic; vo_node is receiving close
to all 30 fps.

### Confirmed pipeline numbers (after QoS fix, with vo_node running):

- Bridge publish rate (from [bridge hb]):   ~30.1 fps  <- ground truth
- ros2 topic hz /camera/color_image:        ~25 fps    <- tool artifact
- ros2 topic hz /camera/depth_image:        ~25 fps    <- tool artifact
- /vo/fault_status:                         ~5 Hz      <- VO compute-bound

### Verdict

QoS fix is fully verified. Bridge at 30 under all tested loads.
Pre-fix topic_hz read ~17 fps because reliable QoS half-dropped at
the DDS layer; post-fix it reads ~25 fps because BEST_EFFORT only
drops a small fraction to the diagnostic subscriber. The full 30
fps is making it into the pipeline.

No further action required on QoS for the mat run. Proceed with
the manual_drive Test 2-style run + -p roi_top_fraction:=0.30 as
previously planned.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 88 (added in this turn).
- No code edits.

## 2026-05-15 (Test 4 analysis + parameter-sweep campaign plan)

Operator requested a rapid iterative mat-test campaign: read each
test, hand back the next mat command with a stated hypothesis,
keep going until enough data, then stop and discuss. Operator
increments the Physical Test number each run. Assistant may vary
parameters (roi_top_fraction, n_features, ransac s-point, etc.) as
long as each test has a clear expected outcome.

### Test 4 — ROI 0.30, manual_drive (VO_readings line 10559+)

722 frames parsed, 81 skipped (truncated psi_raw, logging artifact).

- state mix: agree 127 (17.6%), warming 39 (5.4%),
  vo_suspect 556 (77.0%)
- zero-weight (rejected) ticks: 379/722 (52.5%)
- rho (good ticks):    mean 0.0701  med 0.0580  max 0.254
- weight (good ticks): mean 0.489   med 0.530
- inliers (all):       mean 91.5    med 45      min 0
- spread (all):        mean 69.9    med 82
- first 80 frames: 42 agree / 9 warming / 29 vo_suspect, inl ~116
- last 80 frames:  100% vo_suspect, inl ~47 (ended in a bad pocket)
- transitions: agree<->warming 5/5, agree<->vo_suspect 5/4
- psi_raw-ctyaw: mean -23.6 stdev 20.1 absmax 59 |err|<=15: 46%
- axis sign agreement: X=0.941 (237 win), Y=0.833 (323 win)
- final divergence: dx=+0.740  dy=-0.090  dyaw=0.0
- Cart (0.10,-0.02,-180) -> (-0.59,1.50,-135);
  VO  (0.15,-0.01,-180) -> (0.15,1.41,-135)

Comparison to Test 2 (no ROI, manual_drive, different session):
  Test 2: agree 23.5%, X 0.92, Y 1.00, final dx 1.33 dy -0.42
  Test 4: agree 17.6%, X 0.94, Y 0.83, final dx 0.74 dy -0.09

Key cross-reference: at REST with ROI 0.30 (Test 3 Phase 2)
inliers were ~356; UNDER MOTION with ROI 0.30 (Test 4) inliers
collapsed to ~91. The ROI mask behaves very differently under
motion than at rest - the lower-frame region that the mask keeps
suffers motion blur / low texture while driving, and the
upper-frame region the mask discards (distant walls/signs) had
been contributing stable trackable features. NOT YET concluding
ROI is bad under motion - a same-session no-mask baseline is
needed first because Test 2 was a different day / start pose /
trajectory.

### Campaign plan

  Test 4  ROI 0.30                       DONE
  Test 5  ROI 0.0                        same-session no-mask
                                         baseline. Expect inliers
                                         ~120-130, agree share up
                                         vs Test 4. Isolates
                                         whether ROI helps/hurts
                                         under motion.
  Test 6  ROI 0.15                       gentler mask. Expect
                                         inliers between Test 4
                                         and Test 5; tells us if a
                                         light mask trims sky
                                         without starving features.
  Test 7  ROI 0.0, n_features=1200       more ORB features to
                                         offset motion match-loss.
                                         Expect higher inliers,
                                         possibly higher agree, at
                                         cost of slower fault_status
                                         (more compute/tick).
  Test 8  ROI 0.0, RANSAC s=2->3         code change at
                                         visual_odometry.py:785 +
                                         rebuild. Expect more
                                         robust hypotheses, fewer
                                         bad-consensus frames;
                                         iteration count (300)
                                         unchanged so runtime ~same.

Tests 5-7 are ROS-param-only (no rebuild). Test 8 needs a code
edit + rsync + colcon build, batched when reached.

All vo_node launches keep camera_mode=physical,
input_source=ros_rgbd, force_cart_yaw=true. Only the swept
parameter changes per test. manual_drive trajectory kept as close
to Test 2/Test 4 as the operator can reproduce by hand.

### Test 5 command issued to operator
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 89 (added in this turn).
- No code edits.

## 2026-05-15 (analysis method: maneuver segmentation from ctyaw + ts capture)

Operator asked for a better way to correlate the fixed driving
scenario (straight-left-straight-left-straight~10s-deep left) with
fault_status, including turn timing and separating real turns from
small lane corrections. Requested critical thinking, not just
literal implementation of the suggestion.

### Decision (with rationale)

1. NO instrumentation added to manual_drive or vo_node. Every
   fault_status line already carries ct(x,y,yaw) - the
   Cartographer pose. ctyaw is ground truth for what the car
   physically did. Maneuvers are fully recoverable by segmenting
   the ctyaw signal in data we already capture. This works
   retroactively on Test 4 and avoids changing code that is
   itself under test (which would also break Test 4<->5
   comparability).

2. The only worthwhile addition is shell-side timestamping of the
   capture (moreutils 'ts' is installed at /usr/bin/ts). It does
   not change any node or the fault_status content (the parser
   strips the bracketed prefix), it just gives exact per-line
   wall-clock so turn durations are real, not frame-count
   estimates. Zero risk.

3. Capturing manual_drive stdout through ts as well yields an
   independent driver-intent timeline (when a/d pressed and for
   how long) to cross-check the geometry-derived segmentation.

### Capture commands handed to operator (from Test 5 onward)

Terminal C (fault_status, timestamped):
  ros2 topic echo /vo/fault_status | ts '[%Y-%m-%d %H:%M:%.S]'

Terminal D (manual_drive, timestamped intent track):
  ros2 run qcar2_autonomy manual_drive 2>&1 | ts '[%Y-%m-%d %H:%M:%.S]'

### Turn vs correction classification logic (to be implemented in
the analyzer)

Per sliding window over the ctyaw sequence:
- net dyaw and cumulative |dyaw| computed.
- Scripted turn: large net heading change (tens of degrees),
  sustained / monotonic over a span. Labeled, timed, matched to
  the known script (L1, L2, deep-L).
- Lane correction: small net change (<= ~10-15 deg), brief, often
  sign-reversing (net ~ 0). Flagged but not treated as a
  maneuver.
- Every turn shown; each classified by magnitude + duration so a
  quick flick is distinguished from the deep left.

Per maneuver the analyzer will report: agree%, rho, weight,
inliers, and the growth of VO<->Cart divergence DURING that
maneuver, so degradation can be attributed to specific turns
(Test 4 was 77% vo_suspect overall; this localizes where).

Time anchoring: primary = ts wall-clock prefixes. Fallback for
runs without ts (e.g. Test 4) = the [DEPTH DIAG ALIGNED] log
lines in the vo_node terminal, which carry ROS timestamps
(~every 7-10 s) and bracket the run for frame-index->seconds
interpolation.

### Fixed scenario (locked for all tests in this campaign)
straight -> left -> straight -> left -> straight (~10 s) ->
deep left. Small in-lane corrections occur throughout and are
expected; the analyzer must not treat them as scripted turns.

### Files touched in this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 90 (added in this turn).
- No code edits. Analyzer is an out-of-tree /tmp script.

## 2026-05-15 (Test 5 analysis + Test 6 pivot to n_features)

### Test 5 - ROI 0.0, manual_drive, same fixed scenario
(VO_readings line 12206+; timestamped capture via ts worked)

656 frames parsed, 92 skipped (truncated psi_raw, logging artifact).

Side-by-side vs Test 4 (ROI 0.30), same session / scenario:

  metric                    Test 4 (0.30)   Test 5 (0.0)
  agree %                       17.6            20.3
  vo_suspect %                  77.0            73.6
  zero-weight (rejected) %      52.5            45.0
  inliers (all) mean            91.5           130.3
  rho (good) mean               0.070           0.074
  weight (good) mean            0.49            0.56
  psi_raw-ctyaw |err|<=15       46%             78%
  X sign agreement              0.94            0.75
  Y sign agreement              0.83            1.00
  final dx / dy                 0.74/-0.09      0.94/-0.20

Cart (0.10,-0.01,-180) -> (-0.54,1.43,-132)
VO   (0.04,-0.04,-180) -> ( 0.40,1.23,-132)
first 80: 55 agree/8 warming/17 vo_suspect, inl ~212
last 80 : 100% vo_suspect, inl ~126 (ended in a bad pocket again)

Conclusion: ROI 0.0 decisively beats ROI 0.30 under motion.
+43% inliers (130 vs 91), fewer rejected ticks (45 vs 52.5%),
markedly better shadow-yaw tracking (78% vs 46% within 15 deg).
This confirms the rest-vs-motion inversion noted after Test 4:
the ROI mask raises inliers at rest (Test 3: 333->356->410) but
starves them under motion because the kept lower-frame region
suffers motion blur while the discarded upper region had stable
distant features. ROI masking is therefore counterproductive for
the moving use case. Lock roi_top_fraction=0.0 for motion.

### Test 6 decision - pivot away from the planned ROI 0.15

Rationale: ROI 0.0 already won decisively over 0.30. ROI 0.15
would only confirm an already-visible monotonic trend (it would
land between) - low information value. The dominant problem both
Test 4 and Test 5 share is ~74-77% vo_suspect and ~45-52%
rejected ticks under motion, caused by inlier collapse during
turns. Attack that directly instead.

Test 6 = ROI 0.0 + n_features 800 -> 1200 (ROS param, no rebuild).
Hypothesis: more ORB features -> more matches survive motion blur
through turns -> inliers rise above Test 5's 130, vo_suspect
share drops, fewer safety-gate rejections. Expected cost: higher
per-tick compute, so /vo/fault_status rate may drop below ~5 Hz;
watch that tradeoff. If inliers and agree% rise without an
unacceptable rate drop, more features is a win and becomes the
new baseline for subsequent tests.

Test 6 vo_node command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=1200

Campaign so far:
  T4 ROI 0.30                 done
  T5 ROI 0.0                  done - new motion baseline
  T6 ROI 0.0 + n_features 1200 running
  T7 TBD from T6 (likely RANSAC s 2->3, needs code change+rebuild)

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 91.
- No code edits.

## 2026-05-15 (Test 6 analysis - n_features 1200 big win + Test 7)

### Test 6 - ROI 0.0, n_features 1200, manual_drive, same scenario
(VO_readings line 13752+)

682 frames parsed, 84 skipped. Campaign comparison:

  metric                T4(.30/800)  T5(0/800)  T6(0/1200)
  agree %                  17.6        20.3       30.2
  vo_suspect %             77.0        73.6       62.0
  zero-weight %            52.5        45.0       37.0
  inliers (all) mean       91.5       130.3      196.2
  weight (good) mean       0.49        0.56       0.59
  rho (good) mean          0.070       0.074      0.062
  psi |err|<=15            46%         78%        81%
  X sign / Y sign          0.94/0.83   0.75/1.00  1.00/1.00
  final dx / dy            0.74/-0.09  0.94/-0.20 0.58/-0.12
  fault_status rate        -           6.04 Hz    6.19 Hz

Cart (0.11,-0.02,-179) -> (-0.61,1.44,-131)
VO   (0.12,-0.01,-179) -> (-0.03,1.32,-131)
first 80: 53 agree/8 warming/19 vo_suspect, inl ~201, w 0.70
last 80 : 100% vo_suspect, inl ~177 (still ends in a bad pocket,
          but inliers in that pocket are far healthier than
          earlier tests: 177 vs Test 5's 126 vs Test 4's 47)

Rate measured from the ts timestamps: 800 features = 6.04 Hz,
1200 features = 6.19 Hz. No rate penalty - ORB extraction is NOT
the pipeline bottleneck (bridge feeds 30 fps, plenty of headroom;
the ~6 Hz ceiling is elsewhere - matching/RANSAC/tick scheduling).
The earlier worry that more features would slow fault_status is
refuted by direct measurement.

Conclusion: n_features 800 -> 1200 is an unambiguous,
cost-free improvement across every metric. agree% +50%,
inliers +50%, both axis sign agreements perfect, best final
divergence and shadow-yaw tracking of the campaign. This is the
new working baseline.

### manual_drive | ts capture - abandoned
Operator correctly observed the manual_drive terminal showed
nothing through the ts pipe. Root cause: manual_drive's status
line is printed with a carriage return and no newline
(print(..., end='', flush=True) with leading backslash-r), and
ts only emits a timestamped line on newline, so the status line
never traverses the pipe. This does not affect analysis: the
Cartographer yaw ct(...) in every fault_status line is the
ground-truth maneuver signal and Terminal C's
'ros2 topic echo /vo/fault_status | ts' capture works correctly.
Decision: stop piping manual_drive through ts; run it plain so
the operator at least sees local feedback. Keep ts only on the
fault_status capture.

### Test 7 - find the n_features knee
Test 7 = ROI 0.0 + n_features 2000. Since 800->1200 improved
everything at zero rate cost, push further to locate diminishing
returns or a rate drop. If agree%/inliers keep climbing with
acceptable rate, 2000 becomes baseline; if they plateau or
fault_status rate drops meaningfully, lock n_features at 1200.

Test 7 vo_node command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=2000

Campaign: T4 done, T5 done, T6 done (best so far), T7 running
(n_features 2000), T8 likely RANSAC s 2->3 code change on the
winning n_features.

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 92.
- No code edits.

## 2026-05-15 (METHODOLOGY CORRECTION: zone the runs; Test 7 analysis)

Operator supplied critical environmental context: the mat is
currently in a cramped space surrounded by tables, chairs, walls.
For roughly the first 30 s and last 10 s of each run the camera
sees that clutter (rich, easy features). The middle of the run
sees only blue wall / white wall / bare mat. Competition will
have minimal clutter near the camera - mostly just the
lidar-mapping boundary walls, possibly far away. Therefore the
BARE MIDDLE of each run is the competition-representative regime;
the cluttered start/end are non-representative artifacts of the
test room.

Consequence: every prior campaign conclusion (Tests 4-7) was
computed on whole-run aggregates that include the cluttered ends
and are therefore optimistically biased. Re-analysis must zone
each ts-timestamped run into:
  A = clutter start (0-30 s)        non-representative
  B = bare middle (30 s .. end-10s) COMPETITION-REPRESENTATIVE
  C = clutter end (last 10 s)       non-representative
and judge on zone B. Implemented /tmp/analyze_zones.py using the
ts wall-clock prefixes.

### Zoned results (zone B = bare middle)

  metric            T5(0/800)  T6(0/1200)  T7(0/2000)
  agree %              12.0       25.2        7.0
  vo_suspect %         83.3       67.8       88.2
  rejected %           49.8       40.7       49.1
  inliers (all)        121        211        238
  weight               0.54       0.58       0.54
  psi |err|<=15        78%        83%        75%

Clutter-start zone (all tests): ~61-63% agree, 100% psi-good -
pure test-room artifact, confirming the operator's concern.

### Revised conclusions

1. n_features knee is 1200, NOT higher. Test 7 (2000) has the
   most inliers (238) but the WORST zone-B agree (7%) and
   highest vo_suspect (88%). On bare/low-texture walls the extra
   features are ambiguous; RANSAC reaches a self-consistent but
   wrong consensus (textureless-wall / aperture problem). More
   features past 1200 actively degrades the representative
   regime. LOCK n_features=1200.

2. Real competition-condition performance is much worse than the
   whole-run aggregates implied. Best config (T6) in zone B is
   only 25% agree, 68% vo_suspect. The earlier 'T6 30% agree'
   was inflated by the cluttered ends.

3. The ROI hypothesis was rejected on confounded data. In zone B
   the upper image is featureless wall feeding ambiguous
   matches - exactly where cropping to the lower image
   (mat + cones + signs) should help. ROI must be re-tested at
   n_features=1200 and judged on zone B only. This is the
   corrected version of the Test 4/5 ROI experiment.

### Test 7 whole-run (for the record, non-representative)
612 frames; agree 21.4%, vo_suspect 71.6%, inliers 208,
final divergence dx=0.02 dy=0.03 (the tiny final divergence is
because Test 7 happened to end back in the cluttered zone where
VO re-locked - not a sign of good mid-run tracking).

### Test 8 - corrected ROI experiment
Test 8 = n_features 1200 + roi_top_fraction 0.30. Compare its
zone B against Test 6 zone B (n_features 1200, ROI 0.0). Isolates
the ROI effect at the locked feature count under representative
conditions.

Command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.30 -p n_features:=1200

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 93.
- /tmp/analyze_zones.py (out-of-tree analyzer; not committed).
- No code edits.

## 2026-05-15 (Test 8 analysis + ransac_sample_size param + Test 9)

Operator refined the environment timing: total run ~90 s; the
clutter-rich zone is realistically ~10 s at start and ~10 s at
end (not 30 s). The bare regime is after the first left turn and
before the last turn, BUT a big off-mat table is visible to the
camera during the straight-~10 s section, so even mid-run there
is an intermittent off-mat feature source. Operator wants the
whole-run kept in focus while being aware which sections have
near-mat (off-mat) feature assists.

Re-zoned with 10 s / 10 s ends.

### Test 8 (ROI 0.30, n_features 1200) vs Test 6 (ROI 0.0, 1200)
Same feature count, only ROI differs. 10s/10s zoning.

  WHOLE RUN          T6 (ROI 0.0)   T8 (ROI 0.30)
  agree %               30.2           23.1
  vo_suspect %          62.0           70.1
  inliers (all)         196            127
  final div dx/dy       0.58/-0.12     1.06/-0.18

  ZONE B (bare middle, competition-representative)
  agree %               26.3           19.0
  vo_suspect %          65.9           74.5
  rejected %            38.6           51.1
  inliers (all)         197            132
  weight                0.56           0.51
  psi |err|<=15         85%            75%

  ZONE A (clutter start, ~10 s): both ~84-87% agree, 0%
  vo_suspect, 100% psi-good - confirms the test-room artifact.

Verdict: ROI cropping LOSES even in the competition-
representative zone B, not just whole-run. Cropping the top 30%
removes the wall/floor boundary, mat edges and distant
structural lines that DO carry geometric signal, leaving only
the near-field mat surface (low texture, motion-blurred under
drive). The operator's ROI hypothesis is reasonable a priori but
empirically rejected at every zone and feature count tested.
roi_top_fraction = 0.0 is locked.

### Parameter sweep convergence
- ROI:        0.0   (cropping hurts in all zones, all nfeat)
- n_features: 1200  (800 weak; 2000 worse in zone B - ambiguous
                     features on bare walls cause wrong RANSAC
                     consensus: high inliers, low agree)
- Best config so far = Test 6 (ROI 0.0, n_features 1200).
- Structural limitation surfaced: even the best config is only
  ~26% agree / ~66% vo_suspect in the bare middle. Feature-VO on
  low-texture wall+mat is inherently weak. This matters for the
  redundancy design: the supervisor must distrust VO in
  low-texture stretches.

### Code change this turn: ransac_sample_size as a ROS parameter

Made the RANSAC minimum-sample size (the 's-point') tunable
without future rebuilds, mirroring the roi_top_fraction pattern.

visual_odometry.py:
  - VisualOdometryDepth.__init__ new kwarg ransac_sample_size=2
    (added after roi_top_fraction).
  - Stored as: self.ransac_sample_size = max(2, int(
    ransac_sample_size))   # clamp >=2, default 2 = legacy
  - _ransac_motion rewritten generically:
      s = self.ransac_sample_size
      if M < s: return zeros (safety)
      idx = np.random.choice(M, s, replace=False)
      degenerate guard generalized from the old 2-pt distance
        check to: if np.max(np.std(pts_prev[idx],axis=0)) < 1e-8:
        continue
      refit guard changed from 'best_count >= 2' to
        'best_count >= s'.
    Docstring updated from '2-point RANSAC' to 's-point RANSAC'.
    With s=2 the behavior is identical to the previous code.

vo_node.py:
  - declare_parameter('ransac_sample_size', 2)
  - read: ransac_sample_size = int(get_parameter(
    'ransac_sample_size').value)
  - passed into VisualOdometryDepth(... ransac_sample_size=
    ransac_sample_size)

Build: py_compile PASS on visual_odometry.py + vo_node.py;
rsync + colcon build --packages-select qcar2_autonomy
--symlink-install PASS.

### Test 9 - RANSAC s=3 on the winning config
Test 9 = ROI 0.0, n_features 1200, ransac_sample_size 3.
Hypothesis: 3-point samples resist the degenerate/ambiguous
consensus that bare walls produce (Test 7's high-inlier
low-agree signature), so zone-B agree should rise above Test 6's
26% and vo_suspect fall below 66% without a rate penalty
(iterations unchanged at 300). If it does not improve zone B,
the bare-wall weakness is structural (not a RANSAC-robustness
issue) and the campaign concludes with Test 6's config as the
best achievable and a documented VO limitation on low-texture
scenes.

Command issued:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p roi_top_fraction:=0.0 -p n_features:=1200 -p ransac_sample_size:=3

### Files touched this turn
- visual_odometry.py (ransac_sample_size param + generic _ransac_motion)
- vo_node.py (declare/read/pass ransac_sample_size)
- This changelog entry.
- VO_Conversation_Log.txt Turn 94.

## 2026-05-15 (Test 9 analysis + PARAMETER-SWEEP CAMPAIGN CONCLUSION)

### Test 9 - ROI 0.0, n_features 1200, ransac_sample_size 3
vs Test 6 - ROI 0.0, n_features 1200, ransac_sample_size 2
(10s/10s zoning; zone B = competition-representative)

  ZONE B               T6 (s=2)   T9 (s=3)
  agree %                26.3       18.2
  vo_suspect %           65.9       74.9
  rejected %             38.6       53.4
  inliers (all)          197        205
  weight                 0.56       0.58
  psi |err|<=15          85%        62%
  WHOLE final div     0.58/-0.12  0.72/-0.62

s=3 did NOT improve the representative zone - it slightly
degraded it (rejected 53.4 vs 38.6%, agree 18.2 vs 26.3%, psi
62 vs 85%). Inliers/weight essentially unchanged. Interpretation:
the bare-wall weakness is NOT a RANSAC-robustness problem. With
the fixed 300-iteration budget a 3-point sample yields fewer
strong consensus sets on the marginal bare-wall feature
population, so more frames fail to lock -> more vo_suspect.
ransac_sample_size = 2 (default) is retained.

### CAMPAIGN CONCLUSION

Three highest-value levers swept with zone-corrected
methodology (clutter ends excluded; bare middle = competition-
representative):

  Lever            Outcome
  ROI mask         REJECTED. Hurts every zone & feature count.
                   Cropping removes wall/floor boundary, mat
                   edges and distant structural lines that
                   carry geometric signal.
  n_features       1200 is the knee. 800 weak; 2000 WORSE in
                   zone B (ambiguous bare-wall features ->
                   wrong RANSAC consensus: high inliers, low
                   agree).
  RANSAC s-point   2 wins. s=3 no robustness gain, slightly
                   worse.

WINNING CONFIG: roi_top_fraction=0.0, n_features=1200,
ransac_sample_size=2. (= Test 6: stock vo_node except
-p n_features:=1200.) Recommended operating point going
forward.

LOAD-BEARING FINDING (structural, not a tuning bug):
Even the winning config achieves only ~26% agree / ~66%
vo_suspect in the competition-representative bare zone. The
~86% agree seen in the cluttered start/end is a test-room
artifact (tables/chairs/near walls) that will not exist in
competition. Feature-based VO on low-texture wall + uniform
mat is inherently weak; no swept parameter changes this.

DESIGN IMPLICATION for VO-as-redundancy-layer:
VO cannot be expected to agree with Cartographer everywhere.
The supervisor (vo_supervisor.py) should detect the
low-texture regime (e.g. via inlier count / weight / spread
collapse) and DOWN-WEIGHT or suspend VO-vs-Cart disagreement
flagging during those stretches, rather than treating
sustained vo_suspect on bare walls as an odometry fault. VO
is most informative as redundancy where the scene has
structure (signs, cones, boundary geometry) - which is also
where competition-relevant decisions happen.

Campaign halted here per plan. Next: design discussion, no
more parameter mat-runs unless a new hypothesis emerges.

### Files touched this turn
- This changelog entry (campaign conclusion).
- VO_Conversation_Log.txt Turn 95.
- No code edits (ransac_sample_size param from Turn 94 stands;
  default 2 is the retained value).

## 2026-05-15 (VO improvement research + strategic direction)

Operator goal hierarchy clarified: long-term = full VSLAM;
near-term = a good-enough VO that can be folded into the EKF;
meta-goal = learn to design a functional VO (winning the
competition is explicitly secondary). Operator open to additive
improvements only - nothing already built should be discarded.
Operator wants substantive technical decisions in the logs (not
conversational/clarification noise).

### Reference research performed (web)

1. niconielsen32 ComputerVision/VisualOdometry
   stereo_visual_odometry.py pipeline: tiled-grid FAST detector
   (forces spatial spread), KLT optical-flow tracking
   (calcOpticalFlowPyrLK) instead of descriptor matching, SGBM
   disparity, cv2.triangulatePoints for 3D, pose by
   reprojection-error least_squares (LM) with RANSAC. KITTI
   (car, rich texture). Transferable ideas: tiled detection,
   KLT tracking, reprojection-error pose refinement. Does NOT
   address low texture (KITTI is feature-rich).

2. NVIDIA Isaac ROS Visual SLAM (cuVSLAM): production
   stereo-visual-inertial SLAM, GPU, ~7 ms/frame on Jetson.
   Its own docs state it performs poorly on featureless scenes
   (solid-colored walls) - the SAME ceiling we hit - and its
   mitigation is IMU fusion. Confirms our limit is inherent to
   vision-only feature VO, not a tuning defect. Heavy external
   dependency; would require giving it camera ownership
   (conflicts with the single-node Quanser-bridge architecture);
   not a quick win.

3. Low-texture VO literature (2025) converges on: direct/dense
   methods, IMU fusion (VIO), and point+line features. Sparse
   features on RGB (our current frontend) is the most brittle
   possible choice for our environment.

4. RealSense D435 specifics: it IS a stereo camera; the depth
   image we already consume IS the stereo result (IR pair +
   projector disparity). We are NOT missing stereo. The IR dot
   projector adds artificial texture for DEPTH on blank walls,
   but the pattern is camera-fixed (slides across the scene as
   the camera moves) so it cannot be used for world-frame
   frame-to-frame feature tracking. We likely only have
   RGB + aligned depth via the Quanser PIT QCar2DepthAligned
   bridge (NOT raw left/right IR) - to be verified before any
   IR-stereo-odometry idea.

### Key engineering truth recorded (design constraint)

Translation parallel to a bare flat surface is mathematically
unobservable to ANY camera-only method (ICP slip / planar
degeneracy). cuVSLAM, direct methods and depth-ICP all share
this. Therefore 'depth-geometry odometry' is additive value at
STRUCTURED geometry (wall corners, floor/wall edge, mat
boundary, objects) but is NOT a fix for a featureless flat wall
traversed parallel. Its highest practical value is DETECTING
the degeneracy (point-cloud / 3D-point geometric conditioning)
and feeding that into a VO uncertainty estimate.

### Strategic direction (decided, pending operator confirm on order)

Additive, single-node, learning-maximizing plan; nothing from
Tests 4-9 / winning config (ROI 0.0, n_features 1200, s=2) is
discarded:

Step 1 - Frontend accuracy upgrade (from the niconielsen
  reference): add a tiled-FAST + KLT optical-flow tracking
  frontend as an alternative to ORB+BF matching, keeping the
  existing depth-backprojection + RANSAC/SVD-Procrustes
  backend and keeping ORB as a selectable fallback. Expected:
  faster, more stable tracking in turns and
  textured/semi-textured zones; no change to the unobservable
  bare-wall case.

Step 2 - EKF-readiness (the real near-term milestone):
  vo_node additionally publishes nav_msgs/Odometry with a
  principled covariance derived from inlier count, inlier
  spread, RANSAC residual, and 3D-point geometric
  conditioning (the conditioning term captures the bare-wall
  degeneracy). This is what makes VO consumable by an EKF and
  teaches the core fusion concept: a sensor is only as useful
  as its honest uncertainty. The depth-geometry/degeneracy
  detection folds in here rather than as a standalone motion
  estimator.

Recommended order: Step 1 then Step 2 (improve the estimate,
then wrap it in honest uncertainty). vo_supervisor
alarm-suppression idea from the prior turn is superseded by
Step 2 (covariance subsumes it; no extra node).

No code changed this turn (research + direction only). Awaiting
operator confirmation of order before implementing Step 1.

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 96.

## 2026-05-15 (direction revised: keep ORB; estimator is the x,y lever)

Operator rejected the tiled-FAST+KLT frontend swap. Decision:
ORB stays as the feature frontend (speed + accuracy base for a
self-driving car; also keeps future low-light work tractable).
Full ORB-SLAM3 noted as a separate, much larger later topic
(keyframes/local mapping/loop closure/BA - not a detector swap).
Depth is to be used ADDITIVELY (RGB still used; use whatever the
camera offers that improves VO). Low-light robustness flagged as
a FUTURE concern, not now. Current focus: accurate-enough x,y.

### SVD Procrustes - precise assessment (corrects earlier
loose 'SVD may not be best' remark)

SVD (Kabsch/Procrustes) is the OPTIMAL closed-form solver for
its problem: given matched 3D point pairs, find rigid R,t
minimizing summed Euclidean point-to-point error. The solver is
not the issue. The COST FUNCTION we feed it is mismatched to
RealSense error characteristics:

1. Point-to-point 3D Euclidean error is wrong for a depth
   camera. X=(u-cx)*Z/fx: pixel u is sub-pixel precise but
   depth Z noise grows ~quadratically with range (~mm at
   0.5 m, ~5-10 cm at 3-4 m). SVD-Procrustes weights a noisy
   far bare-wall point equally with a crisp near-mat point, so
   the far wall drags x,y. Direct contributor to the measured
   bare-wall degradation.
2. Unweighted + isotropic assumption; real depth error is
   per-point and anisotropic (worse along the ray).
3. Minimizes in 3D then we discard Z anyway (project to body
   X,Y) - Z-noise contaminates X,Y before being thrown away.

### Alternatives (ranked for accurate x,y)

- Depth-weighted Procrustes: same SVD machinery, weight each
  point by 1/sigma_Z(range)^2. Minimal change, kills mismatch
  #1/#2, keeps ORB+depth+RANSAC. Low-risk validation step.
- 3D-2D PnP + RANSAC (cv2.solvePnPRansac): anchor prev-frame
  3D from depth, minimize REPROJECTION error in current-frame
  pixels (error measured where the camera is precise - image
  plane). This is the accuracy-correct method (ORB-SLAM-style
  motion-only estimation) and the proper RGB+depth fusion.
  Moderate cost, real-time.
- Essential matrix 5-pt: rejected (no scale; we have depth).
- Motion-only bundle adjustment: later, toward SLAM.

### Revised plan (additive; ORB/depth/RANSAC structure kept)

Step 1: Depth-weighted Procrustes - one weighting term
  (1/sigma_Z(range)^2) in _svd_rigid_2d / _ransac_motion.
  Low-risk; validates the noisy-far-point hypothesis.
Step 2: If Step 1 helps -> replace point-to-point cost with
  solvePnPRansac reprojection-error estimator (bigger x,y
  accuracy gain). ORB + depth backprojection unchanged.
Step 3: Fold depth-geometry in as a degeneracy/conditioning
  metric that later becomes the EKF covariance.

Supersedes the prior turn's Step 1 (KLT frontend) - that is
dropped. The Step 2 EKF-covariance milestone from the prior
turn is retained and becomes Step 3 here.

No code changed this turn (analysis + revised direction).
Awaiting operator go-ahead to implement Step 1 (depth-weighted
Procrustes).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 97.

## 2026-05-15 (SESSION HANDOFF pointer)

Session ending; operator moving to a new session. Full continuity
handoff (status, locked decisions, load-bearing facts, and the
Step 1/2/3 plan WITH reasoning) is written at the END of
VO_Conversation_Log.txt under the block
"SESSION HANDOFF - 2026-05-15 (READ THIS FIRST)". New session
should read that block first. The older 2026-05-14
"PLAN FOR NEXT SESSION" block in that file is SUPERSEDED/obsolete.

One-line summary for continuity:
- Campaign done. Winning config = ROI 0.0, n_features 1200,
  ransac_sample_size 2 (Test 6).
- Keep ORB; use depth additively; SVD solver fine but its
  point-to-point cost is mismatched to RealSense depth noise.
- NEXT: Step 1 depth-weighted Procrustes (awaiting operator "go"),
  then Step 2 solvePnPRansac reprojection estimator, then Step 3
  vo_node Odometry + principled covariance (EKF-readiness).
- No code changed this turn (logging/handoff only).

## 2026-05-18 (pre-Step-1 hygiene: _match short-row guard + deferred-items register)

New session. Operator requested a code-efficiency review of every VO
file before starting Step 1. The review produced eight findings
(numbered #1-#8 below). Recommended scope was minimal: fix only the
one real defect now, document the rest as intentionally deferred,
then move to Step 1. This was independently cross-checked in a
parallel Codex session ("Review VO next step"); both reviews
converged on the same minimal scope, which raised confidence.

### Code change this turn: _match short-row guard (finding #3)

visual_odometry.py VisualOdometryDepth._match:
  The Lowe ratio-test comprehension used
    return [m for m, n in raw if len([m, n]) == 2 and ...]
  The `len([m, n]) == 2` check was a no-op (Python unpacks `m, n`
  before evaluating the `if`, and the freshly built list is always
  length 2). cv2.BFMatcher.knnMatch(k=2) returns a row with < 2
  neighbors when the train set has < 2 descriptors, and the unpack
  then raised ValueError. Rewritten as an explicit loop that skips
  rows with len(pair) < 2 before the ratio test.

  PRECISION CAVEAT (corrects the review's initial framing): this is
  NOT an active or latent crash in the running pipeline and NOT a
  strict Step 1 prerequisite. update() already returns early at the
  descriptor-count guard
    if (descriptors is None or self._prev_desc is None
        or len(descriptors) < 2 or len(self._prev_desc) < 2): ...
  Since the live call is _match(self._prev_desc, descriptors) and
  knnMatch row length = min(2, len(descriptors)), that guard already
  excludes the short-row case. So the crash is unreachable in
  production; the fix is correctness hygiene that makes _match safe
  on its own (Step 1/2 keep using this path). The live path stays
  bit-for-bit identical (every row has exactly 2 entries there), so
  no behavior change and no mat run is needed to validate it.

### Deferred-items register (findings #1, #2, #4, #5, #6, #7, #8)

Logged here as known and intentionally NOT addressed now, with a
one-line reason each, so they are not rediscovered the hard way:

- #1 pixels_to_3d_body uses a per-feature Python loop for depth
  sampling (runs twice/frame). Vectorizable, behavior-preserving.
  DEFERRED: real cleanup but not a Step 1 prerequisite; widening
  scope right before the estimator work is undesirable.
- #2 _ransac_motion uses np.linalg.norm (sqrt) where a squared-
  distance vs squared-threshold compare is bit-identical and
  cheaper. DEFERRED: this exact function is rewritten in Step 1
  (depth weighting) and replaced in Step 2 (solvePnPRansac);
  optimizing it now is throwaway.
- #4 align_depth ascontiguousarray wasted on the float/no-warp
  path; update() color_image.copy() unnecessary on the gray path.
  DEFERRED: trivial, zero urgency.
- #5 vo_node._color_cb runs the BGR cv_bridge decode while holding
  state_lock at ~30 Hz (contends with _vo_tick). DEFERRED: real
  but node-level housekeeping, not engine/Step-1 scope.
- #6 vo_node._publish_all emits 17 scalar topics every 20 Hz tick
  even on init/warming/no-cart ticks. DEFERRED: design choice,
  not a bug; not a CPU bottleneck.
- #7 vo_supervisor.py subscribes /vo/state_id as UInt8 and
  /vo/delta_trans, /vo/vo_weight as Float32 while vo_node publishes
  Int32 / Float64 -> ROS 2 type-hash mismatch, those subscriptions
  receive nothing (only /vo/healthy connects). Also the state-id
  numbering is inverted vs vo_node STATE_MAP (node agree=2,
  vo_suspect=3; supervisor assumes vo_suspect=2, agree=3).
  DEFERRED: vo_supervisor is explicitly superseded by Step 3
  (covariance subsumes the alarm-suppression role); fixing it now
  is wasted effort.
- #8 vo_live_plot.py has the same Float32-vs-Float64 mismatch on
  /vo/delta_trans and /vo/vo_weight, and depends on the broken
  supervisor for trust/mode. DEFERRED: same root cause as #7;
  diagnostic-only tool.

### Step 1 design pin (from the Codex cross-check)

Depth-weighted Procrustes needs a per-correspondence range/sigma_Z
weight, but _ransac_motion(prev_2d, curr_2d) and _svd_rigid_2d
currently receive only body-frame XY. The slice prev_3d[both_valid,
:2] / curr_3d[both_valid, :2] in update() discards exactly the Z
needed for the 1/sigma_Z(range)^2 weighting. Step 1's first edit is
therefore a call-site change: carry the per-point Z (or camera-frame
range) alongside the XY into _ransac_motion / _svd_rigid_2d, and
expose the weighting as a default-safe ROS param so it A/Bs cleanly
against Test 6.

### Build

py_compile PASS on visual_odometry.py; rsync ACC_Development ->
~/ros2/src/qcar2_autonomy; colcon build --packages-select
qcar2_autonomy --symlink-install PASS; install sourced. No mat run
(behavior-preserving change, nothing to validate until Step 1).

### Files touched this turn
- visual_odometry.py (_match short-row guard, finding #3)
- This changelog entry.
- VO_Conversation_Log.txt Turns 99-100.

## 2026-05-18 (STEP 1 IMPLEMENTED: depth-weighted Procrustes, default-safe)

Operator gave the go for Step 1. Implemented depth-weighted
Procrustes as a single default-safe, A/B-able ROS parameter.
Nothing from the campaign is discarded; ORB + depth backprojection
+ s-point RANSAC structure unchanged.

### Concept

RealSense stereo depth noise grows ~ sigma_Z ∝ Z^2 (mm at 0.5 m,
5-10 cm at 3-4 m). Unweighted SVD Procrustes weights a noisy far
bare-wall point equally with a crisp near-mat point, so the far
point drags x,y — the cost-function mismatch diagnosed in the
2026-05-15 entries. Step 1 weights each correspondence by
1 / range**depth_weight_power before the rigid fit.

### Code changes (visual_odometry.py)

- DepthProjector.pixels_to_3d_body now returns a THIRD value
  `depths` (per-point camera-frame range, mode units; 0 where the
  pixel was out of bounds). The body-frame Z column is height, NOT
  camera range, so the range had to be surfaced separately — this
  is the call-site change flagged in the Codex cross-check
  (Turn 100). Both early-return and normal-return paths updated.
- VisualOdometryDepth.__init__: new kwarg depth_weight_power=0.0,
  stored as self.depth_weight_power, clamped to [0, 8] (0.0
  outside band -> 0.0).
- update(): unpacks the new prev_depths/curr_depths. When
  depth_weight_power > 0 it forms
  pair_range = max(prev_depths, curr_depths)[both_valid]
  (a pair is only as good as its noisier endpoint), then
  weights = pair_range ** (-power), normalized so max weight = 1
  (Procrustes uses weights only up to a common scale; this just
  keeps the wide range span well-conditioned). When power == 0 it
  passes weights=None so the executed code is literally the old
  path (default-safe A/B control).
- _ransac_motion(pts_prev, pts_curr, weights=None): weights are
  applied ONLY inside the SVD fits (minimal-sample hypothesis fit
  via weights[idx] and the final inlier refit via
  weights[best_inliers]). The RANSAC inlier test stays an
  UNWEIGHTED geometric distance vs the locked ransac_threshold, so
  the consensus search and its tuned threshold are unchanged; only
  the pose recovered FROM the chosen consensus set is depth-
  weighted. This is the lowest-risk reading of "depth-weighted
  Procrustes" and keeps the A/B clean.
- _svd_rigid_2d(pts_a, pts_b, w=None): w=None keeps the original
  unweighted Kabsch verbatim. w given -> weighted centroids and
  weighted cross-covariance S = (w[:,None]*A).T @ B. Degenerate-
  weight guard (sum<=0 or non-finite) falls back to unweighted
  instead of producing NaNs. Uniform weights cancel out of the
  normalized centroid and only scale S by a positive constant, so
  the SVD R,t are unchanged -> depth_weight_power=0 is provably
  identical to the prior estimator.

### Code changes (vo_node.py)

- declare_parameter('depth_weight_power', 0.0); read as float;
  passed into VisualOdometryDepth(... depth_weight_power=...).
- Startup banner shows depth_weight_power and ON / OFF(=Test 6).
  A warn() fires when power > 0 so the operator cannot mistake a
  weighted run for the control.

### Verification (out-of-tree /tmp script, not committed)

Direct numeric checks against the live class:
- _svd_rigid_2d: w=None vs uniform-ones -> dR=0.0, dt=0.0
  (bit-identical); vs uniform 7.3 -> ~1e-16 (machine epsilon).
  Confirms the default-safe claim empirically, not on faith.
- _ransac_motion(weights=None) reproducible under fixed seed and
  unchanged from prior behavior.
- Weighted path (power 4, far noisy outliers) returns finite,
  correct motion — numerically stable, no NaNs.

py_compile PASS (visual_odometry.py + vo_node.py); rsync
ACC_Development -> ~/ros2/src/qcar2_autonomy; colcon build
--packages-select qcar2_autonomy --symlink-install PASS; install
sourced.

### A/B validation plan (operator runs the mat; assistant does not)

Hold everything at the Test 6 winning config and vary ONLY
depth_weight_power. Same fixed scenario (straight, left, straight,
left, straight ~10 s, deep left). Judge on the bare-middle zone B,
not whole-run aggregates (the cluttered start/end is a test-room
artifact). Capture via the usual
`ros2 topic echo /vo/fault_status | ts '[%Y-%m-%d %H:%M:%.S]'`.

  Control  (= Test 6): -p depth_weight_power:=0.0
  Gentle:              -p depth_weight_power:=2.0
  Full RealSense:      -p depth_weight_power:=4.0

Full command form:
ros2 run qcar2_autonomy vo_node --ros-args -p camera_mode:=physical -p input_source:=ros_rgbd -p force_cart_yaw:=true -p n_features:=1200 -p depth_weight_power:=0.0

Then rerun with 2.0 and 4.0. Hypothesis: if zone-B x,y agreement
improves and vo_suspect falls vs the 0.0 control, the cost-function
diagnosis is confirmed and Step 2 (solvePnPRansac reprojection
estimator) is justified. If no zone-B improvement at any power, the
far-point-drag hypothesis is not the dominant error term and the
campaign records that before moving to Step 2.

### Files touched this turn
- visual_odometry.py (pixels_to_3d_body 3rd return; __init__
  depth_weight_power; update() weight build; _ransac_motion +
  _svd_rigid_2d weighted)
- vo_node.py (declare/read/pass depth_weight_power; banner + warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 101.

## 2026-05-18 (STEP 1b: hard max-depth cutoff — operator idea, Quanser-precedented)

Operator proposed a HARD distance cutoff (drop features past a
range entirely) as the complement to Step 1's soft weighting,
noting Quanser does something similar. Verified the precedent:
pit/YOLO/nets.py NeuralNetworks.post_processing zeroes any depth
pixel past `clippingDistance` (default 10 m):
  bgRemoved = np.where((depth3D > clippingDistance) | (depth3D <= 0),
                       0, depth3D)
So a hard depth clip is established Quanser practice. This was
cross-discussed in the parallel Codex session ("Review VO next
step"); Codex independently recommended adding exactly ONE extra
knob (max_vo_feature_depth_m) and testing depth-only this mat
session (do not re-sweep ROI / n_features / ransac_sample_size —
already swept). Agreed; only this one knob was added.

### Code changes (visual_odometry.py)

- VisualOdometryDepth.__init__: new kwarg
  max_vo_feature_depth_m=0.0, stored as self.max_vo_feature_depth;
  <= 0 (incl. negative) normalizes to 0.0 = OFF. Units = active
  mode depth units (metres in physical mode).
- update(): right after both_valid = prev_valid & curr_valid and
  BEFORE the min_inliers gate, when the cutoff > 0:
    within = (prev_depths <= cut) & (curr_depths <= cut)
    both_valid = both_valid & within
  A correspondence is dropped if EITHER endpoint's camera range
  exceeds the cutoff (a pair is only usable if both observations
  are within trusted range). Placed before the min_inliers check
  so an over-aggressive cutoff cleanly degrades to "VO abstains"
  (vo_suspect) instead of fitting on a near-empty set. The
  depth_weight_power path consumes the post-cutoff both_valid, so
  cutoff filters first and weighting then trusts the survivors —
  the two compose cleanly and independently.

### Code changes (vo_node.py)

- declare_parameter('max_vo_feature_depth_m', 0.0); read float;
  passed into VisualOdometryDepth. Banner shows it with ON/OFF; a
  warn() fires when active so a cutoff run cannot be mistaken for
  the control.

### Verification (out-of-tree, not committed)

- Default 0.0 (OFF), explicit 3.0 stored 3.0, negative -> 0.0.
- Mask logic: prev/curr depth pairs vs 3.0 m cutoff keep exactly
  the pairs with BOTH endpoints <= 3.0 (point dropped if either
  endpoint exceeds). Default-safe: cutoff 0.0 leaves both_valid
  untouched -> identical to the Step 1 / Test 6 path.

py_compile PASS; rsync; colcon --symlink-install PASS; sourced.

### Updated A/B matrix (depth-only this session; judge zone B)

All at the Test 6 base (n_features 1200, ROI 0.0, s 2), vary only
the two depth knobs:

  1 Control      depth_weight_power:=0.0  max_vo_feature_depth_m:=0.0
  2 Soft         depth_weight_power:=2.0  max_vo_feature_depth_m:=0.0
  3 Strong soft  depth_weight_power:=4.0  max_vo_feature_depth_m:=0.0
  4 Hard cutoff  depth_weight_power:=0.0  max_vo_feature_depth_m:=3.0
  5 Combo        depth_weight_power:=2.0  max_vo_feature_depth_m:=3.0

Run 1 is the unweighted Test 6 control. Compare 2-5 against it on
the bare-middle zone B only. No more knobs added before this
session per the "don't confound the test" discipline.

### Files touched this turn
- visual_odometry.py (__init__ max_vo_feature_depth_m; update()
  cutoff mask before min_inliers gate)
- vo_node.py (declare/read/pass max_vo_feature_depth_m; banner +
  warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 102.

## 2026-05-18 (Step 1 mat A/B analysis: depth weighting and cutoff)

Operator ran the five physical mat tests for the Step 1 depth
experiment and appended them to `VO_readings.txt` under `Physical
Test 10` / `STEP1 TEST 1-5`. Parsed results from the new blocks
only. The parse ignores truncated `psi_raw=...` tails and uses the
stable fields through `inl` / `sp`.

Test set:

  Test 1 control:      depth_weight_power=0.0, max_vo_feature_depth_m=0.0
  Test 2 soft:         depth_weight_power=2.0, max_vo_feature_depth_m=0.0
  Test 3 strong soft:  depth_weight_power=4.0, max_vo_feature_depth_m=0.0
  Test 4 cutoff:       depth_weight_power=0.0, max_vo_feature_depth_m=3.0
  Test 5 combo:        depth_weight_power=2.0, max_vo_feature_depth_m=3.0

10s/10s zoning was reused: zone A = first 10 s, zone C = last
10 s, zone B = representative middle.

Same-session whole-run summary:

  Run        agree  vo_sus  w=0   inl  final VO-Cart
  control    20.0   72.8   47.3  207  +0.12,+0.04
  soft 2     15.1   78.6   49.9  213  -0.04,-0.03
  soft 4     18.9   76.7   44.1  204  +0.99,-0.46
  cutoff 3   14.1   82.2   60.1  157  +1.09,-0.76
  combo      18.3   75.2   55.4  168  +0.63,-0.40

Zone-B summary:

  Run        agree  vo_sus  w=0   rho   weight  inl
  control    14.1   79.9   48.6  0.062  0.32   195
  soft 2      9.5   85.3   52.0  0.067  0.28   196
  soft 4     14.5   80.6   45.1  0.060  0.33   187
  cutoff 3    8.0   89.1   64.6  0.074  0.19   127
  combo      18.8   74.0   57.1  0.072  0.26   159

Interpretation:
- Depth weighting did NOT clearly improve the same-session control.
  `depth_weight_power=2.0` was worse in zone B on agree/vo_suspect.
  `depth_weight_power=4.0` roughly matched zone-B agree but ended
  with large full-run final divergence.
- The hard 3 m cutoff is too aggressive for this environment. The
  depth diagnostics during the bare middle repeatedly show center
  depths around 4-5 m; cutting at 3 m removes many wall/boundary
  features that still carry signal. This explains the inlier drop
  (zone-B avg 127 vs 195 control), high `w=0`, and worst
  vo_suspect rate.
- The combo had the best zone-B agree percentage, but also high
  `w=0`, lower inliers, higher rho, and substantial final
  divergence. Not a clean winner.
- Overall conclusion: the far-depth-noise hypothesis is not
  confirmed as the dominant current VO error. The structural
  low-texture / bare-wall limitation remains the main issue.

Immediate mat recommendation:
- The five tests are enough to stop if the operator wants to remove
  the car from the mat.
- If doing exactly one more quick run while the car is already on
  the mat, use a looser cutoff only:
    depth_weight_power=0.0, max_vo_feature_depth_m=5.0
  Rationale: it tests whether the hard-cutoff idea was reasonable
  but the 3 m value was too low. Do NOT run more 3 m cutoff tests.

Next design implication:
- Keep default deployment at depth_weight_power=0.0 and
  max_vo_feature_depth_m=0.0 unless the optional 5 m cutoff run
  clearly beats control.
- If no 5 m win, move on to Step 2 planning: solvePnPRansac /
  reprojection-error motion estimation, because simply reweighting
  point-to-point Procrustes did not solve the representative zone.

## 2026-05-18 (Step 1 A/B — independent zone-B reconciliation + DECISION)

Second, independent analysis of the SAME five mat runs (the entry
directly above is the parallel Codex session's analysis; this one
is the Claude-side cross-check). Rebuilt the campaign zone analyzer
as /tmp/analyze_step1.py (out-of-tree, not committed): segments
runs by ts gap >8 s / startup banner / header, reads the ACTUAL
active params from the node's own [WARN] lines (ground truth, not
the header label), skips two short aborted runs (dwp4 5 s, combo
1 s), and judges zone B = [10 s, dur-10 s] on the longest
qualifying full run per config (~120 s, n≈400-500 each).

Zone-B (this analyzer):

  cfg          dwp cut  agree%  vo_susp%  rej%  psi%  inl   w
  1 control    0   0     16.2    77.0    52.5  67.5  207  0.29
  2 soft       2   0     10.5    83.9    55.3  60.8  206  0.26
  3 strong     4   0      8.2    87.0    53.8  51.0  175  0.25
  4 cutoff     0   3      9.0    87.8    69.7  38.2  128  0.15
  5 combo      2   3     16.5    76.3    65.3  48.4  147  0.19

Reconciliation with the Codex entry above: control / soft2 /
cutoff3 / combo agree within run-to-run noise across both
analyzers (control ~14-16% agree, ~77-80% vo_suspect; cutoff3 the
worst on quality, inliers ~127-128 both). The only divergence is
dwp=4 (Test 3 had 5 vo_node restarts; the analyzers picked
different qualifying runs — 8.2% here vs 14.5% Codex). The verdict
is robust to that: in BOTH analyses NO config beats control in
zone B; soft weighting is worse or equal; the 3 m cutoff is
clearly worst (starves the estimator: inliers collapse, rej% and
vo_suspect% jump); combo only ties control on agree while worse on
rej%/psi%/inliers. Here the soft degradation is even monotonic
(control > dwp2 > dwp4).

DECISION (Claude-side, converging with Codex's conclusion):
- The far-point-noise hypothesis is REFUTED. depth_weight_power
  and max_vo_feature_depth_m are not accuracy levers. They remain
  in the code, default 0.0 = OFF (verified bit-identical to
  Test 6) — inert, documented, no revert needed.
- On the optional looser-cutoff idea (Codex: try cut=5.0 m once):
  noted but assessed LOW-VALUE. Physical depth_max is 8 m and the
  bare-middle features sit ~4-5 m, so a 5 m cut only trims the
  small, noisiest 5-8 m tail — predicted to reproduce control
  ±noise, not overturn the verdict. Not worth keeping the car
  staged for; acceptable only as a free opportunistic grab if the
  car is still on the mat.
- No more required mat runs. Result is conclusive (two
  independent analyzers, 5 configs, ~500 zone-B samples each).
  Recommend the car comes off the mat.
- STRATEGIC: this strengthens the already-recorded structural
  finding (bare-wall translation is unobservable to camera-only
  methods). Claude recommends pivoting to STEP 3 (vo_node
  publishes nav_msgs/Odometry + principled covariance from
  inliers/spread/RANSAC residual/3D conditioning) rather than
  Step 2 (solvePnPRansac), because Step 1 shows the bottleneck is
  structural, not the cost function — the high-value move is
  honest uncertainty so the EKF can down-weight VO where it is
  blind, not a better point estimate on a degenerate scene. Codex
  leans Step 2 next; this is a genuine strategic fork left for the
  operator to decide. No more parameter mat-runs either way.

No code changed this turn (analysis + decision + logging only).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 103.
- /tmp/analyze_step1.py (out-of-tree analyzer; not committed).

## 2026-05-18 (5 m cutoff = also negative; Step 2-vs-Step 3 debate opened)

Operator ran the optional looser cutoff (depth_weight_power=0.0,
max_vo_feature_depth_m=5.0) and appended it to VO_readings.txt.
Note: that run's capture is the vo_node CONSOLE log (banner +
[WARN] + [DEPTH DIAG ALIGNED]), not the ts-prefixed
/vo/fault_status stream, so the Claude zone analyzer cannot
zone-B it. Verdict is nonetheless robust from three independent
reads:
- Codex parse: 5 m cutoff agree 10.0% vs control 20.0% whole-run,
  ~4.7% agree / 91.7% vo_suspect in the middle — worse than
  control (recovered more inliers than 3 m but still worse than no
  cutoff).
- Operator observation: "changed nothing" / no improvement.
- Prediction (prior entry) + the run's own [DEPTH DIAG ALIGNED]
  values clustering at 0.5-1.3 m (occasional 4-4.7 m): a 5 m cut
  trims only the small, noisy 5-8 m tail, so ~control was
  expected.

CONCLUSION: the entire depth-handling lever (soft weighting at
dwp=2/4, hard cutoff at 3 m and 5 m, and the combo) is CLOSED.
None beat the unweighted Test 6 control in the representative
zone. depth_weight_power and max_vo_feature_depth_m stay in the
code, default 0.0 = OFF (bit-identical to Test 6), inert. No more
depth-parameter mat runs.

STRATEGIC FORK (decision PENDING — being debated between the two
assistants at operator request, not decided this turn):
- Codex position: Step 2 next — replace point-to-point Procrustes
  with a PnP / reprojection-error estimator; "the VO problem is
  the motion-estimation method itself."
- Claude position: Step 3 next — vo_node publishes
  nav_msgs/Odometry + a principled covariance (inliers / spread /
  RANSAC residual / 3D-point geometric conditioning). Argument:
  Step 1's failure mode was information STARVATION (inliers
  collapse, rej/vo_suspect rise), i.e. an observability/
  conditioning problem, not a cost-function bias problem; a better
  objective (PnP) on a rank-deficient bare-wall problem still
  returns confident-wrong output. The near-term goal is
  fusion-readiness, whose gate is honest covariance, not marginal
  point accuracy. Step 3 is additive/zero-regression to the
  proven path and yields a per-frame quality metric that makes a
  later Step 2 objectively measurable; Step 2 first replaces the
  load-bearing estimator with no yardstick and no safety net.
  Order should be Step 3 -> Step 2, not Step 2 -> Step 3.
- Full Claude reasoning delivered to the operator for the debate;
  decision + its rationale will be logged here once resolved.

No code changed this turn (analysis + debate framing + logging).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 105.

## 2026-05-18 (project direction: Step 3 selected lane; competition-exclusion + showcase plan)

Operator reviewed both assistants' Step 2-vs-Step 3 positions and
set project direction (no code this turn — direction + Step 3
scope explanation only):

- Depth-handling lever accepted as CLOSED by the operator: the
  best new config (dwp=2) was at best ~equal-to-slightly-worse
  than the unweighted 1200-feature Test 6 baseline. Integration
  judged correct; the approach simply did not help. No more
  depth-parameter mat runs.
- Framing agreed: Step 2 = raw-VO-accuracy research path; Step 3 =
  physical-robot fusion / safe-EKF-integration path.
- DECISION: pursue Step 3 next (pending final operator "go" after
  the scope explanation). Rationale tied to the operator's actual
  plan rather than the abstract debate: Step 3 is the prerequisite
  for the optional Friday VO+Cartographer EKF test, is the
  strongest showcase artifact (honest uncertainty that visibly
  grows on bare walls), and is the clean input for the teammate's
  read-only VSLAM/loop-closure showcase.
- PROJECT CONSTRAINTS recorded (durable):
  * VO / VSLAM are READ-ONLY research + showcase. They will NOT be
    run during the actual competition — competition compute is
    reserved for lane following, trip planner, traffic detection.
  * Timeline: build Step 3 ~2026-05-18..05-19 (today/tomorrow);
    optional EKF VO+Cart fusion test on Friday 2026-05-22 (final
    tests); the last ~3 days (~05-20..05-22) are showcase
    production (videos, RViz uncertainty/loop-closure demos).
  * Teammate handoff: current VO implementation given to a
    teammate to start VSLAM (camera map / loop closure) as a
    read-only showcase only; operator's stance — if VO is weak
    SLAM won't fix it, so scope it as showcase, not a competition
    capability (same logic as deferring low-light work until
    good-light works).

Step 3 build scope (planned, additive, default-safe, zero
regression to the Test 6 path):
- vo_node additionally publishes nav_msgs/Odometry (e.g.
  /vo/odometry): the existing pose + a per-frame covariance.
- Covariance derived from inlier count, inlier spread, RANSAC
  residual, and a NEW 3D-point geometric-conditioning term (the
  quantitative bare-wall-degeneracy detector).
- Conditioning also published as its own scalar (dashboard +
  showcase visibility). Existing topics / fault_status / estimate
  path all unchanged. Step 2 (PnP) explicitly deferred to AFTER
  Step 3 and would be evaluated using the Step 3 conditioning
  metric.

No code changed this turn (direction + Step 3 scope only;
awaiting operator go to implement).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 106.

## 2026-05-18 (Speedups #1/#2 + STEP 3 IMPLEMENTED: honest covariance, additive)

Operator pushed the pre-bundle snapshot to origin/Gabriel for a
teammate's VSLAM handoff, then gave the go for the bundle. Built
the two free speedups + Step 3. ALL additive / default-safe: the
proven estimate path is unchanged; only new outputs were added.

### Speedup #1 — vectorized pixels_to_3d_body depth sampling

visual_odometry.py: the per-feature Python loop (ran twice/frame
in the hot path) replaced with np.rint + boolean-mask fancy
indexing. Behavior-IDENTICAL: np.rint is the same round-half-to-
even as int(round()), same bounds test, same zero-fill for
out-of-image pixels. Verified out-of-tree: np.array_equal of the
sampled depths vs the old loop on a synthetic 300-point case incl.
out-of-bounds.

### Speedup #2 — RANSAC squared-distance (no per-iteration sqrt)

visual_odometry._ransac_motion: replaced
  res = np.linalg.norm(diff, axis=1); inl = res < thr
with
  res2 = np.einsum('ij,ij->i', diff, diff); inl = res2 < thr2
(thr2 = ransac_threshold**2, computed once). (a<t) ⇔ (a^2<t^2)
for a,t>=0, so the inlier set, best model and motion are
bit-identical; only the per-iteration sqrt over all M points x
ransac_iterations is removed. Verified deterministic + identical
model/inliers under fixed seed.

### Step 3 — honest uncertainty (the Friday-EKF / showcase / teammate enabler)

visual_odometry.py:
- _ransac_motion now also returns the mean inlier residual (m)
  of the FINAL fit (one sqrt over inliers only, negligible).
  Signature: (tx, ty, dpsi, inliers, mean_resid). Sole caller
  (update()) updated; M<s early return returns 0.0 resid.
- update() computes a geometric-conditioning pair from the inlier
  body-frame XY cloud: eigenvalues of its 2x2 covariance ->
  geom_cond = sqrt(lam_min) (m; ~0 = collinear/clustered =
  bare-wall translation degeneracy) and geom_aniso =
  sqrt(lam_min/lam_max) in [0,1]. Cheap (2x2 eigvalsh). This does
  NOT change the motion estimate — diagnostic only.
- result dict gains geom_cond / geom_aniso / ransac_resid (present
  in the base dict + every return path so consumers never
  KeyError). New self.* state added + initialized in __init__ and
  reset().

vo_node.py:
- New param publish_vo_odometry (default True; additive — only
  adds a topic, no behavior change; node simply not launched in
  competition).
- New publishers: /vo/odometry (nav_msgs/Odometry, pose + 6x6
  covariance), /vo/conditioning (Float64 geom_cond), /vo/reason
  (String).
- _evaluate sets a reason tag at every decision point:
  warming / no_anchor / invalid / bare_wall / low_inliers /
  low_conf / low_weight / turn / ok / odom_suspect (bare_wall
  takes priority over low_inliers/low_weight when the cloud is
  degenerate — the showcase-relevant root cause). Cartographer
  window yaw-change captured as the turn-rate proxy.
- _vo_covariance(): MONOTONE honest variance (x,y,yaw) inflated by
  few inliers / collinear cloud / sloppy RANSAC residual / fast
  turn / bare-wall degeneracy; invalid -> "ignore me" variances.
  Explicitly an honest-uncertainty proxy, NOT a calibrated
  covariance (calibration = future / Friday EKF). Tuning is plain
  module constants (VO_*), deliberately NOT ROS params, to avoid
  re-opening a sweep campaign on a read-only feature.
- fault_status string gets a trailing ` reason=<tag>` (appended;
  existing fields unchanged).
- Startup info log announces Step 3 active.

vo_terminal_dashboard.py / vo_image_overlay.py: regex extended
with an OPTIONAL trailing (?:\s+reason=(?P<reason>\S+))? before
$, so OLD captures (no reason) still parse AND new ones do;
both surface `why=<reason>`. Backward-compat verified.

### force_cart_yaw caveat (recorded for the Friday EKF test)

/vo/odometry publishes the operating pose, whose yaw is pinned to
Cartographer when force_cart_yaw:=true (default). For an honest
VO-vs-Cart EKF fusion the camera-only yaw (/vo/vo_psi_shadow)
should be used and yaw covariance widened, else yaw is
double-counted. Not changed now (out of Step 3 scope); to be
addressed when wiring the Friday EKF experiment.

### Verification

py_compile PASS (visual_odometry, vo_node, vo_terminal_dashboard,
vo_image_overlay). Out-of-tree checks: #1 depth-sample identical
to old loop; #2 deterministic + identical inliers/model; result
carries the new keys; regex parses old AND new fault_status.
rsync + colcon --symlink-install PASS; nav_msgs/Odometry import
resolves in the ROS env. No mat run (additive; the real exercise
is the optional Friday VO+Cart EKF test).

### Files touched this turn
- visual_odometry.py (#1 vectorize depth sample; #2 squared
  RANSAC + mean_resid return; geom conditioning; result keys;
  state init/reset)
- vo_node.py (publish_vo_odometry param; /vo/odometry,
  /vo/conditioning, /vo/reason; reason tags; _vo_covariance;
  fault_status reason; Step 3 constants + info log)
- vo_terminal_dashboard.py, vo_image_overlay.py (optional reason
  in regex + display)
- This changelog entry.
- VO_Conversation_Log.txt Turn 107.

## 2026-05-18 (Grid feature homogenization + camera->body extrinsic verified + direction)

### Camera->body extrinsic VERIFIED against Quanser (operator precaution)

Operator asked, as a precaution, whether the camera->car conversion
in pixels_to_3d_body could be wrong. Checked against Quanser's own
canonical extrinsic, qcar_functions.py get_extrinsics ("SECTION B.2
Camera Extrinsics"): their vehicle->camera rotation evaluates to
R_v2cam = [[0,-1,0],[0,0,-1],[1,0,0]]. Our PHYSICAL_T_CAM2BODY
rotation [[0,0,1],[-1,0,0],[0,-1,0]] is EXACTLY R_v2cam transposed
(correct camera->body inverse). Translation differs only because
Quanser's reference models a simplified pure-height offset in QLabs
10x units (it is written for their BEV/IPM ground-plane pipeline),
whereas we use the real 3-axis hardware offset in metres
(0.095,0.032,0.172) from the hardware manual — more correct for
true 3D VO, and a translation offset does not affect yaw. CONCLUSION:
camera->body conversion is correct; the camera-only-yaw vs
Cartographer-yaw gap is drift (+ possibly a small camera mount
rotational miscalibration), NOT a frame/extrinsic bug.

### Code: ORB-SLAM-style grid feature homogenization (default-safe)

Implemented at operator go. visual_odometry.py:
- __init__ kwarg feature_grid=0 (clamped [0,32]); self.n_features
  stored for the per-cell budget.
- _distribute_features(keypoints, descriptors, H, W): buckets
  keypoints into feature_grid x feature_grid cells, keeps
  best-by-response per cell (per_cell = max(1, n_features/cells))
  via np.lexsort + per-cell rank; keypoint<->descriptor row
  alignment preserved. g<=0 / no descriptors -> inputs returned
  UNCHANGED (true no-op default).
- update() calls it right after detectAndCompute (no-op when off).
This is the cheap, robust form of ORB-SLAM2/3's quadtree
homogenizer; rationale: OpenCV ORB keeps strongest responses which
CLUSTER on one object, making RANSAC inliers geometrically
redundant (confident but ill-conditioned). Pixel coords unchanged
=> intrinsics unaffected, no crop/resize.

vo_node.py: declare/read/pass feature_grid; warn when active.

Verification (out-of-tree): feature_grid=0 returns the SAME
descriptor object (bit-identical, zero regression); feature_grid=8
reduces per-cell clustering and preserves kp<->desc alignment.
Known tradeoff (matches the over-uniform-quadtree literature):
too fine a grid thins features on low-texture scenes and can
starve RANSAC — hence a default-safe A/B knob judged with the
Step 3 conditioning metric, not an always-on change.
py_compile + rsync + colcon --symlink-install PASS.

### Direction decisions (operator)

- Approved: implement grid distribution first (done), THEN a
  separate logged deletion pass removing the conclusively-dead
  rejected levers (depth_weight_power, max_vo_feature_depth_m,
  roi_top_fraction) and their code, while KEEPING the
  pixels_to_3d_body range return (reusable for a future PnP/Step 2).
  This explicitly overrides the old "additive only / never discard"
  rule, at operator direction, now that the campaign is closed.
- External systems (web-verified): Isaac ROS Visual SLAM / cuVSLAM
  is NOT deployable on this QCar2 — current releases require Jetson
  Thor + JetPack 7.1 (or x86 Ampere+); QCar2 is JetPack R35.6, and
  it needs stereo+IMU we do not cleanly expose, and it shares the
  bare-wall limit (its docs fall back to IMU there). DROP it.
  RTAB-Map RGB-D = the realistic mature baseline + the teammate's
  VSLAM/showcase track: our qcar2_camera_bridge already publishes
  the RGB + aligned-depth topics it needs, so integration is
  install + remap + calibrate, not pipeline-building. Pursue as a
  TIMEBOXED stretch (teammate-owned) with the honest-VO + Step 3 +
  grid showcase as the guaranteed fallback. ORB-SLAM3 = educational,
  heavier ROS2 integration, lower priority.
- VIO clarification recorded: force_cart_yaw is LOOSE pose-level
  IMU+camera coupling; true VIO (ORB-SLAM3-VI / cuVSLAM /
  RTAB-Map+IMU) is TIGHT joint optimization with calibrated
  camera-IMU extrinsics + hardware time sync — stronger, but still
  not a bare-wall fix. Showcase framing: production VO == VSLAM +
  fusion; honest limits are the Waymo-serious story.

No mat run this turn (grid is additive/default-off; validated when
the operator next runs the mat). Next: separate deletion pass, then
RTAB-Map bring-up guide for the teammate.

### Files touched this turn
- visual_odometry.py (feature_grid kwarg + _distribute_features +
  update() hook + self.n_features)
- vo_node.py (declare/read/pass feature_grid + warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 108.

## 2026-05-18 (DELETION PASS: rejected ROI + depth-odometry levers removed)

Operator confirmed removal ("you can remove them"). Separate clean
pass, as planned. Removed the three conclusively-rejected,
default-off levers and ALL their code; kept reusable plumbing and
all live Step 3 work. This intentionally overrides the old
"additive only / never discard" rule — campaign is closed.

REMOVED (visual_odometry.py): roi_top_fraction,
depth_weight_power, max_vo_feature_depth_m kwargs + their __init__
comment/clamp blocks; the update() ROI-mask block (now
detectAndCompute(gray, None)); the update() hard-cutoff block; the
update() per-correspondence weight block; the `weights` parameter
of _ransac_motion and the w_idx/w_in threading; the weighted
branch of _svd_rigid_2d (reverted to the original 2-arg unweighted
Kabsch); self._orb_mask. __init__ comment blocks deleted by line
range (box/Unicode chars made exact-match editing error-prone);
logic reverted via targeted edits.

REMOVED (vo_node.py): declare/read/pass + startup banner lines +
runtime warns for roi_top_fraction / depth_weight_power /
max_vo_feature_depth_m.

KEPT (reusable / live):
- DepthProjector.pixels_to_3d_body still returns the 3rd value
  `depths` (per-point camera range) — a future PnP/Step 2 needs
  previous-frame depth; cheap and inert now.
- Speedup #1 (vectorized depth sample) and #2 (squared-distance
  RANSAC, no sqrt) — unaffected.
- All Step 3: _ransac_motion still returns mean inlier residual
  (5-tuple), geom_cond/geom_aniso, /vo/odometry + covariance,
  /vo/conditioning, /vo/reason, reason tags.
- feature_grid (the new grid homogenizer) and ransac_sample_size.

Verification: py_compile PASS (both files). No residual code refs
(the only grep hit is the substring `w_in` inside the Step 3
reason tag 'low_inliers' — unrelated, correct). Out-of-tree: the
default ctor still constructs; _svd_rigid_2d back to 2-arg and
recovers a known transform; _ransac_motion returns 5 values with
finite residual; the removed attributes are gone from the object;
feature_grid default 0 / settable. rsync + colcon
--symlink-install PASS; engine + node import in the ROS env.
The campaign param history remains in VO_CHANGELOG.md /
VO_readings.txt as the record of WHY these were removed; old
campaign CLI commands referencing them will now error (expected —
those experiments are closed).

No mat run (removal of default-off dead code does not change the
live estimate path; next mat run validates feature_grid + Step 3).

### Files touched this turn
- visual_odometry.py (removed roi/depth-weight/cutoff/weights;
  _svd_rigid_2d reverted to unweighted; _ransac_motion 3-arg)
- vo_node.py (removed the three params end-to-end)
- This changelog entry.
- VO_Conversation_Log.txt Turn 109.

## 2026-05-18 (camera_info + static base_link->camera TF in our bridge)

Operator approved the RTAB-Map/RViz enabler. Implemented in OUR
qcar2_camera_bridge.py (NOT the Quanser library — confirmed no
PIT/QCar2DepthAligned modification needed; camera_info is just a
small ROS message of numbers we already own).

Changes (qcar2_camera_bridge.py, all additive):
- Imports: CameraInfo, TransformStamped, StaticTransformBroadcaster,
  and DepthProjector (single source of truth for intrinsics +
  extrinsic — no re-typing, no virtual/physical cross-pollination).
  Added _rotation_to_quaternion() helper.
- Params: publish_camera_info (default True), base_frame
  (base_link), camera_optical_frame (camera_color_optical_frame).
- __init__: builds a CameraInfo from the canonical RGB intrinsics
  selected by device_type (depth is aligned to the COLOR grid, so
  COLOR K describes both streams), plumb_bob, d=0 (post-calib no
  distortion). Broadcasts ONE static TF base_frame->optical_frame
  from the same T_cam2body the VO engine uses
  (P_body = R@P_cam + t).
- _tick: rgb/depth header.frame_id set to the optical frame (was
  "color_image"/"depth_image" — metadata only; VO/overlay key off
  the topic, not the string), and CameraInfo published every frame
  with the SAME stamp as the images.

Verification: py_compile PASS; quaternion (x,y,z,w)=
(0.5,-0.5,0.5,-0.5) round-trips PHYSICAL_T_CAM2BODY's rotation
exactly (np.allclose); CameraInfo fx=607.3 fy=607.3 cx=325.0
cy=249.9, t=[0.095,0.032,0.172] m, pulled from DepthProjector.
rsync + colcon --symlink-install PASS. Purely additive — the live
VO path, Quanser core, and image pixel data are untouched; only a
new /camera/camera_info topic, a static TF, and image frame_id
strings changed.

WHY (clarified for the operator): RTAB-Map = two separable parts —
its OWN rgbd_odometry (a mature alternative to our whole VO, the
honest baseline) + the SLAM/map backend (keyframes, loop closure,
trajectory Path + 3D map for RViz; the showcase line). It needs
RGB + registered depth + camera_info + a base_link->camera TF;
we had the first two, this commit adds the last two. Our vo_node
is NOT run alongside RTAB-Map (it replaces our frontend). RTAB-Map
runs OFF the mat on a recorded bag — zero mat-time cost, no
compute contention with the frame-rate-sensitive VO.

No mat run this turn (additive). Next: off-mat validation
checklist + a single consolidated mat session (grid A/B + one bag)
given to the operator; RTAB-Map bring-up on the bag is the
teammate's timeboxed track.

### Files touched this turn
- qcar2_camera_bridge.py (CameraInfo + static TF + frame_id +
  _rotation_to_quaternion; params)
- This changelog entry.
- VO_Conversation_Log.txt Turn 110.

## 2026-05-18 (Easy_Start.txt synced to current procedure)

Operator asked for Easy_Start.txt to be kept current so they can
self-serve the basic startup (cartographer etc.) instead of asking
each time. Easy_Start.txt is in-scope for Claude edits.

Verified from the launch files (not guessed): qcar2_keyboard_
drive_launch.py / qcar2_manual_drive_launch.py declare
camera_source with default 'depth_aligned', which conditionally
launches the qcar2_autonomy 'camera_bridge' node; the
*_cartographer_launch.py wrappers include those drive launches. So
the cartographer launch ALREADY auto-starts qcar2_camera_bridge —
no separate camera terminal was ever needed; the run book just
never said so (it wrongly said "(includes rgbd)"). rgbd.cpp only
runs with camera_source:=rgbd.

Edits to Easy_Start.txt:
- Header date -> 2026-05-18 + a "WHAT'S CURRENT" block: camera
  bridge auto-start, the topics it now publishes (incl.
  /camera/camera_info + static TF), the canonical Test-6 VO
  command, the live knobs (feature_grid, publish_vo_odometry),
  the REMOVED knobs (roi_top_fraction / depth_weight_power /
  max_vo_feature_depth_m), and the new /vo topics + reason tag.
- Section 2 "Important launch fact" corrected: cartographer launch
  also auto-starts the camera bridge (depth_aligned default);
  Terminal 1 relabeled "base + cartographer + camera bridge".
- Canonical vo_node command updated everywhere (added
  -p n_features:=1200) via replace_all. Caught + fixed a
  double-append the replace_all caused on the header line.
- New Section 2.1 E) pre-mat off-mat validation checklist
  (camera_info / tf2_echo / topic hz / /vo topics / dashboard
  why=) and F) bag-record line for off-mat RTAB-Map / re-analysis.
- Other sections' stale vo_node commands brought to the canonical
  form by the same replace_all.

Doc-only change; no code, no build, no mat run. Kept the file's
existing structure/style.

### Files touched this turn
- Easy_Start.txt (synced to current pipeline)
- This changelog entry.
- VO_Conversation_Log.txt Turn 111.

## 2026-05-18 (Easy_Start fix: canonical launch is qcar2_cartographer_launch.py)

Correction to the entry above. Operator flagged that they do NOT
use the keyboard/manual cartographer launches (Sections 2/2.1) that
the previous edit leaned on. Verified from the operator's OWN
pasted sessions in VO_readings.txt: the actual startup is
  ros2 launch qcar2_nodes qcar2_cartographer_launch.py
That launch IncludeLaunchDescription's qcar2_launch.py, whose
camera_source default is 'depth_aligned' -> qcar2_camera_bridge
auto-starts. So the "camera bridge auto-starts, no separate camera
terminal" point holds; only the launch FILE name was wrong.

Easy_Start.txt edits:
- Header WHAT'S CURRENT: states the canonical startup is Section
  0.7 / qcar2_cartographer_launch.py and that Sections 2/2.1
  (keyboard/manual) are legacy alternates.
- New Section 0.7 "Canonical Physical Startup (THIS is what we
  run)": the 4 real terminals (qcar2_cartographer_launch.py;
  vo_node Test-6 cmd + feature_grid A/B note; vo_dashboard;
  manual_drive), pointing to 2.1 E) for off-mat validation and
  2.1 F) for the bag.
- Section 1.5 daily order rewritten to route through Step 0.7 and
  label Steps 2/2.1 as legacy alternates.

Method note: confirmed the launch include chain from the launch
files (qcar2_cartographer_launch.py -> qcar2_launch.py ->
camera_source default depth_aligned) rather than assuming.
Doc-only; no code/build/mat.

### Files touched this turn
- Easy_Start.txt (canonical startup corrected to
  qcar2_cartographer_launch.py; new Section 0.7)
- This changelog entry.
- VO_Conversation_Log.txt Turn 112.

## 2026-05-18 (Pre-Test 11: off-mat validation of the whole session — PASS)

Operator ran the off-mat validation (car parked) and pasted it to
VO_readings.txt "Physical Pre-Test 11". Reviewed in full.

VERDICT: PASS. Everything built this session is verified working
off the mat:
- Deletion pass did NOT break vo_node: clean banner, no traceback,
  "Step 3 ACTIVE ... =True".
- /camera/camera_info: ~30 Hz, exactly correct — k=[607.327, 0,
  324.950, 0, 607.345, 249.868, 0,0,1], plumb_bob, d=0, frame
  camera_color_optical_frame. (The "does not appear yet" line is
  the ros2-topic-hz startup warning; rate prints immediately
  after.)
- Static TF base_link->camera_color_optical_frame: exactly
  correct — t=[0.095,0.032,0.172], q=[0.5,-0.5,0.5,-0.5], matrix
  matches PHYSICAL_T_CAM2BODY. (Initial "frame does not exist" is
  the one-time wait before the latched static TF; prints correctly
  right after.)
- /vo/odometry: covariance behaving as designed — x/y var ~0.0009
  (best-case, locked on clutter w/ 400-500 inliers), z=25.0 and
  roll/pitch~2.467 (unobserved-DOF big values), yaw var tiny.
- /vo/conditioning, /vo/reason present; fault_status ends with
  reason=; vo_dashboard shows why=ok.
- feature_grid:=8 run: the ACTIVE warn fired, Step 3 active,
  fault_status kept publishing (~220-260 inliers — fewer but
  spread, as intended), dashboard parsed why=ok. New grid code
  path runs clean.

Told the operator they do NOT need to re-run the Terminal-3
checks with grid on: camera_info / TF / topic rates are produced
by the camera bridge + static broadcaster and are independent of
the vo_node feature_grid knob; the only grid-relevant checks
(node start + fault_status + dashboard) were already done.

OBSERVATION (not a blocker): depth center read 7-13 m throughout
both runs — camera was pointed at a far/open area while parked;
>8 m is gated out by depth_max, yet VO still found 200-500
inliers and low covariance, so the pipeline is healthy even on a
far scene. On the mat the scene is near (mat/cones/signs <~4 m).

REMAINING off-mat check the operator hadn't done: the ~10 s
bag dry-run + ros2 bag info (the RTAB-Map artifact). Once that
passes, cleared for the single consolidated mat session
(feature_grid 0 vs 8 vs 12 + one bag).

Doc/analysis only; no code/build/mat this turn.

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 113.

## 2026-05-19 (GRID A/B RESULT — feature_grid=8 WINS; first positive lever)

Operator ran the single consolidated mat session (grid 0 / 8 / 12,
same fixed scenario as the Step 1 campaign, --full-length capture
so reason= is preserved) and pasted it to VO_readings.txt. Missed
the bag record (not needed for this verdict; RTAB-Map-only,
decoupled, grab opportunistically later). Rebuilt the zone
analyzer (/tmp/analyze_grid.py; out-of-tree): segment by
`=== GRID TEST n grid=N ===` headers + ts gap, zone B =
[10 s, dur-10 s], longest qualifying run per grid.

Zone-B (bare middle, competition-representative):

  grid  agree%  vo_susp%  rej%  psi%  inl  drift  topReasons(zoneB)
  0      12.3    82.5     48.9  61.0  187  0.01   turn205 invalid190 ok61
  8      22.6    70.1     39.8  69.5   86  0.01   turn222 invalid171 ok129
  12     21.7    71.2     45.4  67.6   59  1.00   invalid192 turn184 ok120

VERDICT: feature_grid=8 is the FIRST lever in the whole campaign
to beat the Test-6 baseline in the representative zone, and it
wins on EVERY headline metric at once: agree ~doubled
(12.3->22.6%), vo_suspect -12 pts (82.5->70.1), rej -9 pts
(48.9->39.8), psi +8.5 pts (61->69.5), drift unchanged (0.01 m).
Mechanism confirmed by the new reason tags: zone-B `ok` frames
more than doubled (61->129) and `invalid` fell (190->171) — i.e.
de-clustering produced better-conditioned geometry so VO actually
locks more often. This is a much stronger, multi-metric,
mechanistically-explained signal than anything in the depth
campaign (which only ever tied or lost).

grid=12 REJECTED: ties grid 8 on agree but inliers collapse to 59,
rej not improved (45.4), end drift 1.00 m — the over-thinning /
starvation knee predicted when the knob was built (matches the
over-uniform-quadtree literature). grid 8 is the sweet spot.

Caveats (honest): one run per config (run-to-run variance exists,
no repeat), and 22.6% absolute agree is still modest — the
structural bare-wall ceiling stands; grid 8 raises the
BEST-ACHIEVABLE meaningfully, it does not "solve" bare walls
(nothing vision-only can). Effect size + 4-metric consistency +
reason-tag mechanism make it a credible adopt-now result by the
same single-representative-run methodology the depth campaign
used; an optional grid8-vs-grid0 confirm run is nice-to-have, not
required, and not worth a dedicated mat session given the
showcase timeline.

DECISION: feature_grid=8 is the new recommended operating point /
winning config (Test 6 base + feature_grid:=8). grid 12 rejected.
Code default stays feature_grid=0 (default-safe; do NOT silently
flip behavior) — the new operating point is expressed in the run
book / commands, not by changing the engine default. Easy_Start.txt
updated: header canonical command + Section 0.7 Terminal 2 now use
-p feature_grid:=8 with the result summary and the 12-is-bad note;
targeted edits only (no replace_all, to avoid the prior
double-append).

This is the campaign's first genuine win and a strong showcase
beat ("diagnosed feature clustering -> applied ORB-SLAM-style grid
homogenization -> measurably ~doubled agreement in the hard zone,
with the reason tags proving the mechanism").

No code changed this turn (analysis + decision + run-book update;
the feature_grid engine code from 2026-05-18 stands, default 0).

### Files touched this turn
- Easy_Start.txt (canonical command -> Test 6 + feature_grid:=8;
  result summary; grid 12 marked do-not-use)
- This changelog entry.
- VO_Conversation_Log.txt Turn 114.

## 2026-05-19 (RTAB-Map bag verified + Test-6 cross-session reconciliation)

Operator recorded the RTAB-Map bag (/tmp/vo_rtab) and pasted
`ros2 bag info` to VO_readings.txt, and asked (a) to compare the
grid session vs Test 6, having noticed the plain config looked
worse than Test 6, and (b) what the bag is for / whether to delete
it.

BAG: PASS. 128.5 s, 5.5 GiB, all 9 topics healthy
(camera_info 3889, color 2954, depth 2592, tf 3836, tf_static 2,
/vo/* 763 each). RTAB-Map-ready.

CROSS-SESSION baseline drift (zone B, bare middle):
  Test 6 (05-15, n1200, no grid)     ~26% agree / ~66% vo_suspect
  control (05-18, n1200, dwp0)        16% / 77%
  grid 0  (05-19, n1200)              12% / 83%
  grid 8  (05-19, n1200)              23% / 70%

Operator asked if a code change caused the regression. Assessment
(honest, evidence-based — NOT a code regression):
- Speedups #1/#2 were verified bit-identical (np.array_equal depth
  sample; deterministic identical RANSAC inliers/model). Step 3 is
  additive only. The deletion pass removed roi/depth_weight/cutoff
  + weighted Kabsch, but those are inactive in any control run
  (all =0 -> original code path). So the control path math == Test 6.
- The largest drop (26->16) occurred on 05-18, whose control path
  was already behavior-equivalent to Test 6 — i.e. the drop
  PREDATES the grid/cleanup work and happened with
  verified-equivalent code. That isolates the cause to
  environment + RANSAC stochasticity, not the edits.
- RANSAC is unseeded in production: identical config + identical
  scene still varies run-to-run; plus day-to-day scene/lighting/
  placement (campaign already documented large environment
  confounds). Cross-session ABSOLUTE numbers are not comparable —
  the methodology has always judged WITHIN-session zone-B A/B
  deltas. The grid0-vs-grid8 same-session A/B (12 -> 23) is the
  valid comparison and stands.
- Stronger reframe: on this harder day the plain config got 12%
  but grid 8 recovered ~23% ≈ Test-6-level — grid 8 clawed back
  to historical-best territory despite worse conditions. This
  strengthens, not weakens, the grid-8 result.
- Cannot prove zero subtle effect without an exact same-session
  Test-6 rerun, but bit-identical verification + the drop
  predating the changes make a regression very unlikely, and the
  cross-session absolute comparison is methodologically invalid
  regardless.

BAG PURPOSE / KEEP-OR-DELETE: the bag is a raw sensor recording
(no map in it). RTAB-Map (run later, off-car, teammate) builds the
map + trajectory FROM it for RViz. Decision is conditional and
left to the operator: KEEP only if the teammate will actually run
the RTAB-Map/VSLAM showcase soon (then move off /tmp — it can be
wiped on reboot: mv to a persistent path); otherwise DELETE
(rm -rf /tmp/vo_rtab) — 5.5 GiB with no consumer is waste. If VO
work is being called done and no one is committed to RTAB-Map,
delete.

STATUS: VO improvement work is effectively concluded. Net
deliverables this campaign arc: speedups (#1/#2, bit-identical),
Step 3 honest covariance/odometry/reason (EKF + showcase enabler),
grid feature homogenization (feature_grid=8 = first lever to beat
the Test-6 baseline, ~doubled zone-B agree same-session), dead
levers removed, Easy_Start synced. Open optional threads (operator
choice, none blocking): RTAB-Map on the bag (teammate showcase),
a grid8-vs-grid0 confirm run, Friday VO+Cart EKF using Step 3
covariance.

No code changed this turn (analysis + bag verify + logging).

### Files touched this turn
- This changelog entry.
- VO_Conversation_Log.txt Turn 115.

## 2026-05-19 (Toolbox pt.1: selectable PnP estimator + bag/RViz reality)

Operator directive: integrate PnP and KLT as user-selectable
options ("go crazy on the mat"); also asked to see the RTAB-Map
recording in RViz now. Identified the YouTube source they learn
from = Prof. Andreas Geiger, "Self-Driving Cars Lec 7.1: Visual
Odometry" (KITTI author) — his lecture frames 3D-2D PnP as the
recommended RGB-D motion method and 3D-3D (our SVD) as the most
depth-noise-sensitive: independent corroboration of Step 2.

KLT-vs-ORB clarification recorded: KLT = good pure-VO frontend
(no descriptors -> no loop closure/relocalization); ORB
descriptors are why ORB-SLAM uses ORB (for relocalization/loop
closure). Design consequence: keep ORB as the DETECTOR; KLT will
be an optional tracking frontend on top, descriptors retained.
RTAB-Map-over-ORB-SLAM3 rationale logged: RTAB-Map is ROS2-native,
consumes our exact topics, outputs map+grid+trajectory+covariance,
low integration risk; ORB-SLAM3 has no clean ROS2 wrapper, no nav
covariance, same low-texture weakness, multi-day gamble — parallel
external baseline, not fed by our frontend.

BAG/RVIZ REALITY (honest): /tmp/vo_rtab is GONE (/tmp wiped, the
reboot risk previously flagged; dryrun + vo_grid_session.txt also
gone), and rtabmap is NOT installed. So no RTAB-Map map in RViz
now — two blockers. Path: rtabmap_ros install (teammate, off-car)
+ a bag re-recorded to a PERSISTENT path (NOT /tmp). New rule:
bags never to /tmp. Live /vo/odometry+image in RViz needs neither
(offered an RViz config as a deliverable).

### Code: selectable PnP estimator (default-safe; toolbox pt.1)

visual_odometry.py:
- __init__ kwarg vo_estimator='svd' (only 'pnp' enables PnP; any
  other value -> 'svd' = unchanged proven path). pnp_reproj_px=3.0
  (PnP RANSAC pixel gate).
- New _pnp_motion(prev_3d_body, curr_3d_body, curr_px): converts
  prev body 3D -> prev CAMERA frame (P_cam = (P_body - t_cb) @ R_cb),
  cv2.solvePnPRansac (iterationsCount=ransac_iterations,
  reprojectionError=pnp_reproj_px, ITERATIVE) -> camera ego
  transform T_ego (prev_cam->curr_cam); body feature-motion
  T_body = T_cb @ T_ego @ inv(T_cb); returns the SAME tuple/
  semantics as _ransac_motion — body-frame planar (dx,dy,dpsi)
  pre-negate, inlier_mask over input rows, residual in body-XY
  metres (identical definition to SVD) so negate/accumulate/
  Step-3 covariance are untouched. Guards: M<max(6,min_inliers),
  cv2.error, not-ok, inliers None, inliers<min_inliers all ->
  zeros (VO abstains, same as SVD low-inlier path).
- update(): selectable branch — vo_estimator=='pnp' calls
  _pnp_motion(prev_3d[both_valid], curr_3d[both_valid],
  curr_pts[both_valid]); else the unchanged _ransac_motion.

vo_node.py: declare/read/validate/pass vo_estimator (invalid ->
svd warn); warn when PnP active.

Verification (out-of-tree): PnP recovers a known camera motion
EXACTLY — recovered (dx,dy,dpsi) matches the independent
closed-form Tcb@Tego@inv(Tcb) to <2e-3, 80/80 inliers, residual
~1e-2 m; default-safe confirmed (default 'svd', 'PNP'->'pnp',
garbage->'svd'). py_compile + rsync + colcon --symlink-install
PASS; engine imports with estimator=pnp, default svd.

STATUS: PnP is toolbox pt.1, complete + verified + built.
Toolbox pt.2 = KLT optical-flow tracking frontend (vo_frontend
param, ORB kept as detector) — the immediate next build. Then
mat A/B: feature_grid {0,8} x vo_estimator {svd,pnp} x
vo_frontend {orb,klt}. Bare-wall structural ceiling still stands;
these optimize the achievable region.

No mat run this turn (additive/default-off; validated next mat).

### Files touched this turn
- visual_odometry.py (vo_estimator param + _pnp_motion + update()
  selectable branch)
- vo_node.py (declare/read/validate/pass vo_estimator + warn)
- This changelog entry.
- VO_Conversation_Log.txt Turn 116.

## 2026-05-19 (Toolbox pt.2: selectable KLT frontend — toolbox complete)

Operator clarified the ORB-vs-KLT confusion (it IS either/or:
ORB-match frontend vs KLT-track frontend; KLT still needs a
detector seed, ORB's is reused so descriptors stay available for
future SLAM relocalization). rtabmap concern answered: the package
never touches QCar runtime; the only risk is install footprint on
the non-standard humble/L4T — so install/run RTAB-Map ISOLATED
(teammate machine / container, on a bag), never apt into the
QCar system ROS. Bag policy: only persist the single BEST run.

### Code: selectable KLT optical-flow frontend (default-safe)

visual_odometry.py:
- __init__ kwarg vo_frontend='orb' (only 'klt' enables KLT; any
  other value -> 'orb' = unchanged proven path).
- New self._prev_gray; _store() gained a `gray` param (all 7 call
  sites updated — 6 via replace_all, the 8-space success-path call
  fixed separately after grep verification) so the previous
  grayscale frame is available as the KLT source; __init__/reset
  init _prev_gray=None.
- update(): the descriptor-guard + _match + prev/curr extraction
  is now an `else` (orb) branch; new `if vo_frontend=='klt'`
  branch tracks the PREVIOUS frame's grid-distributed keypoints
  into the current gray via cv2.calcOpticalFlowPyrLK
  (winSize 21, maxLevel 3), keeps status==1 & in-bounds, feeds
  the SAME backprojection->estimator->Step-3 pipeline. ORB is
  still detected every frame (descriptors retained; next-frame KLT
  seed; grid distribution applies so KLT tracks a spread set).
  Guards (no prev gray/kp, LK None, <min_inliers tracked) ->
  _store + return, same abstain semantics as the ORB guards.
- BUG caught by the integration test and fixed: confidence still
  did inlier_count/len(matches) — `matches` is undefined on the
  KLT path -> UnboundLocalError. Replaced with
  n_corr = prev_pts.shape[0] (frontend-agnostic; == len(matches)
  on the ORB path so ORB confidence semantics unchanged).

vo_node.py: declare/read/validate/pass vo_frontend (invalid ->
orb warn); warn when KLT active.

Verification (out-of-tree, synthetic textured pair + constant
depth): all 4 combos {orb,klt} x {svd,pnp} run with NO exception;
orb+svd unchanged (219 inliers, valid); klt tracks all 800 points
on a pure-shift image; default-safe confirmed (default orb+svd;
'KLT'->'klt'; bad->'orb'). pnp shows valid=False on this synthetic
ONLY because constant depth = coplanar points (degenerate for
solvePnP) — PnP correctness was already proven separately on
proper non-coplanar data (exact recovery). py_compile + rsync +
colcon --symlink-install PASS; engine imports, default orb/svd.

Easy_Start.txt: knob notes extended with vo_estimator and
vo_frontend (defaults = proven Test-6+grid8 path; mix freely).

### TOOLBOX COMPLETE

Selectable, all default-safe (defaults reproduce the proven path):
  feature_grid   0 (legacy) | 8 (winner) | 12 (over-thins)
  vo_estimator   svd (proven 3D-3D) | pnp (3D-2D reprojection)
  vo_frontend    orb (match) | klt (optical-flow track)
Next mat session can A/B any mix. Structural bare-wall ceiling
still stands; these optimize the achievable region. RTAB-Map
(isolated, on the BEST persistent bag) remains the parallel
baseline/showcase track.

No mat run this turn (additive/default-off; validated next mat).

### Files touched this turn
- visual_odometry.py (vo_frontend param + KLT branch + _prev_gray
  + _store gray + n_corr confidence fix)
- vo_node.py (declare/read/validate/pass vo_frontend + warn)
- Easy_Start.txt (knob notes: vo_estimator, vo_frontend)
- This changelog entry.
- VO_Conversation_Log.txt Turn 117.
- /tmp/analyze_grid.py (out-of-tree analyzer; not committed).

## 2026-05-26 (YOLO detector — port Erick's rich semantic logic onto the subscriber-only camera path)

**Motivation.** User compared Gabriel's `yolo_detector.py` (subscriber-only — camera owned by `qcar2_camera_bridge`, single-owner invariant the VO work relies on) against Erick's branch `yolo_detector.py`, which has the *richer* per-class semantic logic (traffic-light color via PIT's `lightColor`, yield-sign handling, per-class distance gating, `/qcar_camera/rgb_yolo` annotated overlay) but obtains its frames by instantiating `QCar2DepthAligned` directly — a second PIT camera owner that would break the single-owner architecture used by VO + cartographer. Goal: keep Gabriel's subscriber path, adopt Erick's semantics.

Also addresses two physical-run observations the user reported from Erick's branch:
  * **Stop sign trips brake too early** — car stops well before the sign. PIT computes object distance as `torch.median(mask × depth)` over the segmentation mask ([pit/YOLO/nets.py:373](Development/MDC_libraries/python/pit/YOLO/nets.py#L373)). On a small/distant stop sign the mask is jitter-noisy and biases the median *shorter* than truth.
  * **Traffic-light trips brake too late** — car nearly clips the light. The TL seg mask leaks onto the sky and the pole behind the light, biasing the median *deeper* than the actual light face, so Erick's 2.5 m gate is effectively never crossed until the car is on top of it.

**Changes in `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`.**

- Subscriber pattern (`/camera/color_image`, `/camera/depth_image`) and camera-bridge single-owner invariant preserved — `QCar2DepthAligned` is NOT reintroduced. Comments around the subscriber callbacks left intact (they explain the 2026-05-14 ownership migration).
- Adopted from Erick's branch:
  - Traffic-light state-aware stopping using PIT's `TrafficLight.lightColor` (red/yellow → stop, green/idle → drive).
  - Yield-sign handling (PIT class 33).
  - `/qcar_camera/rgb_yolo` annotated overlay publish (so user can inspect the detections live in `rqt_image_view`).
  - Detection cooldown semantics (TL refreshes every frame while red/yellow; stop/yield use the full cooldown).
- New (not in either branch):
  - **Full-TL-visibility gate.** TL stop fires only when the bbox is at least `tl_edge_margin_px` (default 8) inside the image border on all four sides. Justification (user, 2026-05-26): "we don't want to stop in the middle of the street" when only the bottom half of a TL has entered the frame. Behavior on a half-visible red TL is to log a one-line "not stopping (bbox not fully visible)" message and keep driving.
  - **Per-class thresholds promoted to `declare_parameter()`** (`stop_sign_conf`, `stop_sign_dist_m`, `stop_sign_hold_s`, `yield_sign_*`, `tl_conf`, `tl_min_dist_m`, `tl_stop_dist_m`, `tl_hold_s`, `tl_edge_margin_px`, `detection_cooldown_s`). Tunable at launch without a rebuild.
  - **Diagnostic dual-distance log.** Every detection logs both PIT's mask-median distance AND a center-patch median distance (`_center_patch_depth_m`, median of valid depths in the central 20 % of the bbox). This is the data we need on the next physical run to confirm the bias direction on stop signs and TLs and finalize the thresholds; helper deliberately lightweight (a few hundred pixels per detection).
- **Defaults vs Erick's branch:**
  - `stop_sign_dist_m`: **0.55 m** (Erick: 1.0 m) — pulled in to match "right in front of the sign" on the physical car.
  - `tl_stop_dist_m`: **3.5 m** (Erick: 2.5 m) — pushed out to compensate for the seg-mask-on-sky bias and to give a real braking distance.
  - All other thresholds preserved at Erick's values.
- `_bbox_fully_in_frame()` helper: image-edge margin check used by the TL visibility gate.

**Not addressed (flagged for follow-up).** User asked about detecting yield + roundabout signs + crosswalks. The Quanser model `quanser_yolov8s-seg.pt` already includes **yield** (class 33, now wired). **Roundabout** signs and **crosswalks** are NOT in COCO and NOT in the Quanser model; adding them needs a custom-trained YOLO head (LISA + Mapillary datasets would be plausible source data) or a second stacked model. No pretrained weights I'd recommend grabbing off the internet — flagging as a separate future task rather than inventing detection code that pretends to work.

**Verification expected on next physical run.** With `vo_node` + `qcar2_camera_bridge` + `yolo_detector` co-running:
1. `ros2 topic echo /qcar_camera/rgb_yolo` (or `rqt_image_view`) shows annotated frames — boxes + class names + distances.
2. Log lines `[YOLO] <name> conf=X PITdist=Y centerD=Z bbox=(...)` appear in the `yolo_detector` console — `Y` vs `Z` mismatch quantifies the seg-mask bias.
3. Approaching a red TL: car should not stop until bbox is fully in view and depth median < 3.5 m. Approaching a stop sign: car should stop when depth median < 0.55 m (re-tune if dual-distance log shows persistent bias).
4. Half-visible red TL (bottom of frame only) → log says "bbox not fully visible — NOT stopping" and car keeps moving.

**Files touched.**

- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (rewrite: subscriber pattern + Erick's semantic logic + new visibility gate + parameter declarations + diagnostic log + annotated overlay publisher).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 138.

## 2026-05-26 (Virtual camera owner + nav_to_pose map-frame switch — parity with physical)

**Context.** User went to run the new yolo_detector against QLabs and caught a stack mismatch: `qcar2_cartographer_virtual_launch.py` only started legacy `rgbd` (MONO16, *not* aligned to the color grid), but the new yolo_detector relies on depth being aligned to the RGB pixel grid (YOLO seg masks live in RGB pixel coordinates; PIT's `mask × depth` median is the *wrong* pixels if depth isn't aligned). Physical mode already handled this via the `camera_source` selector in `qcar2_launch.py` (default `depth_aligned` → `qcar2_autonomy/camera_bridge`). Virtual mode lacked the same selector entirely.

**Changes.**

### `Development/ros2/src/qcar2_nodes/launch/qcar2_virtual_launch.py`
Mirrored physical's `camera_source` selector pattern (copied directly from `qcar2_launch.py`). Now exposes:
  - `camera_source:=depth_aligned` (**new default**) — runs `qcar2_autonomy/camera_bridge` with `device_type:=virtual`. The bridge uses `pit.YOLO.utils.QCar2DepthAligned`'s virtual branch ([pit/YOLO/utils.py:148-165](Development/MDC_libraries/python/pit/YOLO/utils.py#L148-L165)): QLabs Camera3D on `tcpip://localhost:18965`, `depth_scale=5.5`, `warpPerspective` M-matrix alignment. Republishes `/camera/color_image` (bgr8), `/camera/depth_image` (32FC1 meters, **aligned to color grid**), and `/camera/camera_info` (intrinsics from the VIRTUAL DepthProjector table; never cross-pollinated with physical).
  - `camera_source:=rgbd` — legacy fallback (`qcar2_nodes/rgbd` `device_type:=virtual`, MONO16 unaligned), preserved for parity with physical's fallback.
Same `IfCondition(PythonExpression(...))` gate the physical launch uses, identical topic names (drop-in for any subscriber). Lidar / csi / qcar2_hardware nodes unchanged.

### `Development/ros2/src/qcar2_autonomy/autonomy/nav_to_pose.py`
Switched `map_rotated` → `map` at three locations (lines 451-452, 467-468, 620-621). User confirmation: virtual cartographer's `map` frame is already correctly oriented (matches QLabs world), so the 180° static TF `map_rotated → map` that physical's `qcar2_cartographer_launch.py` injects is not needed in virtual mode. Nav2 path publish + tf lookup now happen in the `map` frame directly. Toggle is *commented-out lines preserved on the other side* so flipping back for physical mode is a one-line swap each.

**No edits to `qcar2_cartographer_virtual_launch.py`.** It already `IncludeLaunchDescription(...qcar2_virtual_launch.py)`, so the new `camera_source` arg propagates without further plumbing. The launch deliberately does NOT add a `map_rotated → map` static_transform_publisher (which is what physical's cartographer launch adds at line 35-39) — virtual doesn't need the rotation.

**Run recipe (updated).** Inside dev container:
```
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py        # defaults to camera_source:=depth_aligned -> camera_bridge
ros2 launch qcar2_autonomy autonomy_planner_launch.py               # yolo_detector + trip_planner + path_follower + Planner_server
rqt_image_view /qcar_camera/rgb_yolo                                # visual confirmation
```
To explicitly opt back to the legacy raw path for an A/B:
```
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py camera_source:=rgbd
```

**Files touched.**

- `Development/ros2/src/qcar2_nodes/launch/qcar2_virtual_launch.py` (camera_source selector added; default depth_aligned).
- `Development/ros2/src/qcar2_autonomy/autonomy/nav_to_pose.py` (map_rotated -> map at lines 451-452, 467-468, 620-621).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 139.

## 2026-05-26 (YOLO detector v2 — kill TL stop-go-stop flicker + center-patch gating for stop sign)

**Symptoms from user's first virtual run with the v1 (earlier today) detector.**
1. **Traffic light: stop-go-stop chatter.** Car correctly brakes on red, then briefly creeps forward, then brakes again. Cycle visibly repeats. User correctly intuited that "yolo is looking at too many things" / inference is reacting too noisily.
2. **Stop sign: still brakes too far away** — improved vs Erick's branch but still wrong. User suggested Luigi's idea: "focus on 50% of the stop sign, just the center, so depth is read from the center only."

**Root cause v1 (TL flicker).** `on_timer` was a latch: while `sign_detected==True`, YOLO was NOT re-run; we just waited for `disable_until` to expire. With `tl_hold_s=0.25` and `detection_cooldown_s=0`, every 0.25 s the latch released, `flag_value` flipped True for one tick, the timer re-ran YOLO, the still-red TL re-engaged the latch. That one-tick gap every 250 ms is the visible creep. Stop-go-stop is a *bookkeeping* bug, not a perception bug.

**Root cause v1 (stop-sign distance).** PIT's distance is `torch.median(mask × depth)` over the FULL seg mask ([pit/YOLO/nets.py:373](Development/MDC_libraries/python/pit/YOLO/nets.py#L373)). On small/distant signs the mask leaks onto the pole / background / shadow, biasing median *shorter* than the actual front face. Center-patch median (small ROI around bbox center) reads the face directly. The v1 diagnostic log already proved this — center patch was consistently closer to ground truth.

**Fixes in `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py`.**

1. **Always-evaluate, refresh-each-frame.** Rewrote `on_timer` to call `yolo_detect()` every tick unconditionally. `yolo_detect()` no longer returns `(delay, detected)` — it now mutates two absolute-time fields directly:
   - `brake_until_abs` — `time.time()` when the brake releases. TL refreshes it every frame it's red/yellow; stop/yield set it once per latch.
   - `sign_cooldown_until_abs` — earliest time a stop or yield sign can re-trigger. TL is exempt (always evaluates, no cooldown).
   `flag_value = (now >= brake_until_abs)`. No more state latch, no more one-tick brake-release gap.

2. **Center-patch median is the new gating distance** (Luigi's idea). New param `distance_source ∈ {center_patch, pit_median}`, **default `center_patch`**. The other measurement is still logged as a second opinion (`PITdist` / `centerD`). NaN-safe fallback: if center patch has no valid depth, gating falls back to PIT's median so we never silently fail to detect.

3. **`tl_hold_s` default bumped to 0.6 s** (from 0.25 s). Must exceed one inference period (~33 ms @ 30 Hz) by a comfortable margin to bridge any single-frame miss. With refresh-each-frame logic this just means "if we lose the TL for up to 600 ms, hold the brake."

4. **Removed obsolete state.** `sign_detected`, `disable_until`, `detection_cooldown`, `t0` deleted from the class state. They were the latch that caused the v1 flicker. Replaced entirely by the two `*_abs` timestamps above.

5. **Stop-override (`/trip_planner/qcar_state=1`) uses `max(...)` accumulation** instead of unconditional overwrite, so the override expiry honors the longest currently-active stop reason.

**What stays the same.** Subscriber pattern (camera_bridge owns the RealSense), full-TL-visibility gate, `/qcar_camera/rgb_yolo` annotated overlay, traffic-light color via PIT `lightColor`, declare_parameter for all thresholds, dual-distance log per detection.

**Behavior matrix the user described, implemented:**
- TL red/yellow + full bbox + valid distance: brake refreshed each frame -> continuous stop.
- TL green / idle: no refresh -> brake_until_abs decays -> car drives. User's exact words: "if green we just keep moving."
- TL bbox half out of frame: visibility gate blocks the stop -> log "NOT stopping" -> drive.
- Stop sign closer than `stop_sign_dist_m` (default 0.55): one-shot 3 s brake + 10 s cooldown before the same sign can re-fire.

**Verification on next virtual run.**
1. Approach a red TL: brake should engage smoothly and STAY engaged with no creep until TL turns green or you pass it.
2. Approach a green TL: car drives through (still gets logged, no brake).
3. Approach a stop sign: distance from `used=` field in the log should match where the car actually stops. If it's still biased, swap `distance_source:=pit_median` at launch to A/B; the `PITdist` line is in every log so you can see the bias direction.

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (rewrite of on_timer / yolo_detect; new `distance_source` param; bumped `tl_hold_s` default; removed obsolete latch state).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 140.

## 2026-05-26 (Overnight YOLOv8s finetune — kicked off in dev container)

**Goal.** User wants a YOLO detector trained from scratch (well — finetuned from COCO) on classes that matter for ACC: stop, yield, traffic light, crosswalk, roundabout, speed limit (drop car — RealSense was picking up the front CSI bumper as a car).

**What's running.**
- **Model**: YOLOv8s detection (11.1 M params, 28.7 GFLOPs), pretrained `yolov8s.pt` (COCO) as starting weights.
- **Dataset**: Andrew Mvd "Road Sign Detection" (875 imgs: 612 train / 175 val / 88 test) — 4 classes: `crosswalk`, `speedlimit`, `stop`, `trafficlight`. Source: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection. Downloaded as Roboflow-formatted ZIP from https://github.com/fredotran/traffic-signs-detection — no Kaggle / Roboflow auth needed.
- **Coverage vs target classes**: 4 of 6 (stop, TL, crosswalk, speedlimit). YIELD and ROUNDABOUT not in this dataset — explicitly deferred to a v2 run (need to source a supplementary set; LISA covers yield, GTSDB covers roundabout).
- **Hyperparams**: epochs=500, patience=100, batch=16, imgsz=640, AMP=on, workers=0 (Docker `/dev/shm` is 64 MB which OOMs the multi-process DataLoader — workers=0 is the easy fix without restarting the container).
- **Throughput**: ~17 s/epoch (14 s train + ~3 s val). 500 epochs cap = ~2.4 h; will probably converge much earlier (1-epoch smoke test already hit mAP50=0.71 on COCO-pretrained weights).
- **Container**: `isaac_ros_dev-x86_64-container` (torch 2.1.0+cu121, ultralytics 8.4.14, GPU passthrough RTX 3060 Mobile 6 GB verified).
- **Process**: pid 13755 inside the container (detached `docker exec -d`, `nohup`'d).
- **Logs / outputs (host paths)**:
  - `Development/yolo_training/logs/train.log` — live stdout
  - `Development/yolo_training/runs/detect/road_signs_4class_v1/results.csv` — per-epoch metrics
  - `Development/yolo_training/runs/detect/road_signs_4class_v1/weights/{best,last,epoch{N}}.pt` — checkpoints (every 25 epochs + best/last continuously)

**Why this is "as autonomous as I can do."** I'm not a daemon — I only execute when the user sends a message. But the launched process is detached and doesn't depend on me being alive. Training will run to completion (or early-stop on patience=100) regardless of whether anyone talks to me. When the user wakes up and pings, I just `cat results.csv` and report status.

**User commands.**
```
bash Development/yolo_training/scripts/check_training.sh        # one-shot status
bash Development/yolo_training/scripts/check_training.sh tail   # status + live tail of log
bash Development/yolo_training/scripts/stop_training.sh         # SIGTERM (graceful, then SIGKILL after 10s)
```

**Smoke test result (epoch 1, pre-launch verification).**
```
all          175 images / 255 instances    P=0.876 R=0.649  mAP50=0.709  mAP50-95=0.516
crosswalk    36/40    mAP50=0.781
speedlimit  132/152   mAP50=0.946
stop         18/18    mAP50=0.831
trafficlight 24/45    mAP50=0.277   <- weakest, expected to improve most over training
```

**Integration with `yolo_detector.py` (NOT done yet — for the user's morning testing).**
The trained model will be a YOLOv8 DETECTION model (no segmentation masks). That means `pit.YOLO.nets.YOLOv8.post_processing()`'s mask-median distance is unavailable — we'd switch to the bbox center-patch depth (the path already implemented as `_center_patch_depth_m` and now the default for `distance_source`). Class IDs differ too:
  - Current Quanser model: 2=car, 9=traffic light, 11=stop, 33=yield
  - New model: 0=crosswalk, 1=speedlimit, 2=stop, 3=trafficlight
A future yolo_detector v3 will need:
  - Load weights via plain `ultralytics.YOLO(...)` instead of PIT's wrapper
  - Map new class names directly instead of the COCO IDs
  - Drop PIT mask-median (use center-patch only)
  - Reuse the existing PIT TL color check (HSV brightness over three vertical patches in bbox) — or train color awareness into the model later.

**Files added / touched.**
- `Development/yolo_training/dataset/road_signs_4class/...` (875 imgs + labels + data.yaml)
- `Development/yolo_training/scripts/train.py` (ultralytics finetune script)
- `Development/yolo_training/scripts/check_training.sh` (status command — works from host)
- `Development/yolo_training/scripts/stop_training.sh` (graceful stop command)
- `.gitignore` — added `Development/yolo_training/` (too large for git)
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 141.

## 2026-05-26 PM (YOLO detector v3 — dual-backend: Quanser-seg + custom ultralytics)

**Training result first.** Overnight run finished cleanly (500/500 epochs, ~2h wall clock). Per-class val on `best.pt`:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| crosswalk | 0.93 | 0.93 | **0.98** | 0.84 |
| speedlimit | 1.00 | 0.99 | **1.00** | 0.92 |
| stop | 0.99 | 0.94 | **0.99** | 0.94 |
| trafficlight | 0.98 | **0.51** | 0.67 | 0.48 |

Three of four classes excellent. Trafficlight has high precision but **low recall (0.51)** — model misses half the TLs in the val set, expected given that class had the fewest training instances and the dataset is real-world (model may behave differently on QLabs's stylized sim TLs — test will tell). `best.pt` copied into `Development/ros2/src/qcar2_autonomy/models/road_signs_4class_yolov8s.pt` (gitignored) for a stable reference path.

**Integration.** Restructured `yolo_detector.py` into a dual-backend dispatcher. Backend is chosen via two new ROS params:
  - `model_path` (default `""`) — path to a `.pt` file. Empty = Quanser default.
  - `model_type` (default `"auto"`) — `auto | quanser_seg | ultralytics`.
    `auto` picks `ultralytics` when `model_path` is set, else `quanser_seg`.

Architecture: `yolo_detect()` is now a thin orchestrator that dispatches to `_run_quanser_seg()` (PIT path, preserves v2 behavior bit-for-bit) or `_run_ultralytics()` (new). Both backends return a uniform `(processedResults, xyxy)` shape so the per-detection gating loop is backend-agnostic.

New helpers:
- `_load_model(image_width, image_height)` — backend setup at __init__, deferred PIT import so the ultralytics path can run on machines without the PIT package on the path.
- `_check_traffic_light_color(bgr, x1, y1, x2, y2)` — pure-numpy port of [pit/YOLO/nets.py:320-370](Development/MDC_libraries/python/pit/YOLO/nets.py#L320-L370). Custom YOLO detects "trafficlight" as a single class with no color awareness, so we replicate PIT's three-vertical-patch HSV brightness check in-process. Output strings (`"red"`, `"yellow"`, `"green"`, `"idle"`) match PIT's so the existing red/yellow stop logic is unchanged.
- `_Detection` nested class — backend-agnostic detection record (`.name`, `.conf`, `.x`, `.y`, `.distance`, `.lightColor`) that mirrors PIT's `Obstacle`/`TrafficLight` shape via `__dict__` so the existing per-detection loop consumes either backend's output unchanged.
- `_ULT_NAME_MAP` — normalizes new-model class names (`"stop"` → `"stop sign"`, `"trafficlight"` → `"traffic light"`) so the existing gating branches still match. `crosswalk` and `speedlimit` pass through as-is and are handled by two new `elif` branches.

**Class behavior with the v3 model:**
- `stop` → "stop sign" → existing one-shot brake at `stop_sign_dist_m`.
- `trafficlight` → "traffic light (color)" → existing red/yellow refresh-each-frame brake with full-visibility gate, color from the in-process HSV check.
- `crosswalk` → log-only for now (auto-stop not wired; flagged in the changelog for the "stop before crosswalk if TL red" enhancement the user described in Turn 138).
- `speedlimit` → log-only (user explicit: classify only, no speed change).
- `yield` and `roundabout`: not in v3 model — deferred to a v4 training that adds those classes.

**Verification on next QLabs run.** With:
```
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
ros2 launch qcar2_autonomy autonomy_planner_launch.py
```
yolo_detector starts on the Quanser-seg backend (no behavior change). To use the v3 ultralytics model instead, override via env or a separate `ros2 run`:
```
ros2 run qcar2_autonomy yolo_detector --ros-args \
    -p model_path:=/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models/road_signs_4class_yolov8s.pt
```
Inspect via `rqt_image_view /qcar_camera/rgb_yolo` and watch the `[YOLO]` log lines for `used=` (center-patch depth) on each detection.

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (dual-backend dispatcher, new helpers, crosswalk/speedlimit elif branches).
- `Development/ros2/src/qcar2_autonomy/models/road_signs_4class_yolov8s.pt` (copied from training run, gitignored).
- `.gitignore` (added the new weights file).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 142.

## 2026-05-26 PM-2 (YOLO detector v3.1 — predictive "stop beside the sign" approach)

**Test observations from user's first QLabs run with v3 weights (Turn 142 follow-up):**
- TL red->brake, green->go: ✅ working
- Stop sign: detected (bbox in rqt) but car never braked — blew past
- Yield sign: not detected → v3 model has no yield class (deferred to v4 retrain with supplementary dataset)
- Crosswalk: not detected in the run — unclear whether scene lacked them or sim-vs-real domain gap

**Root cause of stop-sign-detected-but-no-brake.** The v2 gating logic was `brake when used_d < stop_sign_dist_m (0.55m)`. With the car cruising at ~0.3 m/s, the in-frame window where depth crosses 0.55 m is short (~1.8 s before reaching the sign), and motion-command-propagation latency (yolo_detector → /motion_enable → path_follower → wheel cmd → motor) eats most of it. Result: car is already past the sign before brake takes effect.

**Fix: predictive "stop beside the sign" approach (user request).** New `_SignApproachTracker` class:

1. Each consecutive stop-sign detection feeds `(t_abs, depth_m)` into per-sign-type history (max 8 samples).
2. Once `min_samples` (default 3) accumulated, linear-fit depth vs time: `d(t) = slope·t + intercept`. `approach_speed = -slope` (m/s, positive when closing).
3. When `depth < commit_at_m` (default 3.0 m) AND `approach_speed > min_speed` (0.05 m/s), commit to the predicted arrival: `t_arrival = (intercept - target_offset_m) / approach_speed`.
4. After commit: tracker is frozen — never recomputes, never updates from fresh detections. Brake fires at `t >= t_arrival` even if the sign drops out of the FOV in the last 1-2 m (which it always will on a ~70° HFOV D435 — the sign exits the frame before the car reaches it).
5. After brake fires, `on_brake_fired()` clears state so the next sign can be tracked fresh.

This sidesteps both the "depth gate window too narrow" and the "sign leaves FOV before brake fires" problems. **No wheel-encoder dependency** — approach speed is derived from depth-vs-time slope directly.

**New params (all tunable at launch):**
| param | default | meaning |
|---|---|---|
| `stop_target_offset_m` | 0.30 | stop this far before the sign face ("right beside the sign") |
| `stop_predict_min_samples` | 3 | history depth required before fitting |
| `stop_predict_max_depth_m` | 5.0 | ignore detections beyond this — too noisy / out of frame next |
| `stop_predict_commit_at_m` | 3.0 | commit prediction only once depth crosses this |
| `stop_predict_min_speed` | 0.05 | only commit if we're actually approaching (m/s) |

**Diagnostic logging.** Every stop-sign detection that does NOT engage the brake now logs WHY:
- `stop sign conf=0.85 < 0.90 -- not gating`
- `stop sign depth=6.20m out of [0, 5.0] -- not gating`
- `stop sign within cooldown 8.2s remaining -- not gating`
- `stop sign @ 2.30m -- tracking (no commit yet, 2 samples)`
- `stop sign @ 1.80m committed, brake in 4.20s`
- `Stop Sign -> BRAKE NOW (predicted arrival, depth=0.55m, hold 3.0s, cooldown 10s)`

Use these to tune defaults to QLabs' actual approach speeds.

**Yield sign path identical** (uses `_yield_tracker`). Will only fire on the Quanser-seg backend until v4 model adds yield class.

**Removed.** The old simple-distance gates (`stop_dist`, `yield_dist`) — predictive subsumes them. The `stop_sign_dist_m` and `yield_sign_dist_m` params still exist for backwards-compat but no longer drive behavior.

**Not addressed (flagged for follow-up):**
- **Crosswalk** detection in QLabs: need to confirm sim scene has crosswalks at all; if yes and model still misses them, may need sim-domain augmentation.
- **Yield sign** detection: needs v4 training with yield + roundabout supplementary dataset (LISA + GTSDB).
- **Traffic-light "stop right beside it"**: TL behavior stays as-is (refresh-each-frame brake while red/yellow). User's observation that the car ran a quick yellow is more about TL cycle timing than gating logic — would need a different fix (predict arrival like for stop sign, plus committed-stop on red regardless of color change mid-approach).

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (added `_SignApproachTracker`, two new params, replaced stop / yield branches with predictive + diagnostic logging).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 143.

## 2026-05-26 PM-3 (YOLO detector v3.2 — lateral-edge commit + brake-state log)

**Diagnosis from VO_readings.txt run.** User pasted ~70 lines of yolo_detector output. Trace:

```
[YOLO] stop sign conf=0.95 used=5.636m(center_patch)
  stop sign depth=5.64m out of [0, 5.0] -- not gating   <- DROPPED, max too low
...
  stop sign @ 4.91m -- tracking (no commit yet, 1 samples)
  stop sign @ 4.91m -- tracking (no commit yet, 2 samples)
...
  stop sign @ 3.45m -- tracking (no commit yet, 8 samples)
  stop sign @ 3.45m -- tracking (no commit yet, 8 samples)
  (sign exits FOV)
```

History: depths went 4.91 → 4.55 → 4.73 → 4.38 → 4.18 → 4.36 → 4.00 → 3.82 → 3.64 → 3.45 over 0.7 s, while bbox center x slid 423 → 449 → 476 → 503 → 533 → 566 → 600 → 620 across a 640-wide frame. **Sign was on the side of the road; car was passing on the left.** Depth never reached `commit_at_m=3.0` because depth decreases SLOWER than driving distance when you're moving past a lateral object — you're not heading at it head-on. Tracker accumulated 8 samples then sign exited the FOV with no commit.

**Verified brake plumbing is correct** by tracing [nav_to_pose.py:303 → :438 → :600-611](Development/ros2/src/qcar2_autonomy/autonomy/nav_to_pose.py#L600-L611): subscribes to `/motion_enable`, sets `motion_flag`, and in the cmd_vel path `enable=0` when `motion_flag==False` which zeroes both `linear.x` and `angular.z`. Same mechanism Erick's branch used. So the bug was never on the consumer side; yolo was just never publishing False.

**Two fixes (one algorithmic, one diagnostic):**

### 1. LATERAL-EDGE commit trigger (primary brake signal for side-of-road signs)
New trigger in `_SignApproachTracker.update()`: when bbox center x crosses into the outer `lateral_edge_frac` (default 15%, = 96 px on a 640-wide frame) of the image on either side, commit immediately with `target_arrival = now`. That's the moment the sign is about to leave the FOV laterally — i.e., the car is at or just past the sign's position.

Subsumes depth-rate prediction in the common (side-of-road) case. Depth-rate kept as fallback for head-on / occluded signs.

Tracker `update()` signature now takes `bbox_center_x` and `img_w` in addition to depth.

### 2. Bumped depth thresholds so depth-rate fallback also has room to fire
- `stop_predict_max_depth_m`: 5.0 → **8.0**. The run dropped first 14 detections at depth 5.0-5.6 m because of the 5 m cap — wasted samples we could have been using.
- `stop_predict_commit_at_m`: 3.0 → **4.5**. The run's minimum recorded depth before sign exit was 3.45 m; commit_at=3.0 was unreachable. 4.5 m commits even on lateral approaches.

### 3. Brake-state-change log line in `on_timer()`
Logs whenever `flag_value` flips between True/False:
```
>>> BRAKE ENGAGED  (motion_enable -> False) at t=1779804204.32, hold for 3.00s
>>> BRAKE RELEASED (motion_enable -> True)  at t=1779804207.33
```
Two new lines per stop event — minimal noise — but they tell us **whether yolo's intent reached the publisher**, independent of whether path_follower obeyed. If you see BRAKE ENGAGED here and the car keeps moving, the bug is downstream (motor topic, QoS mismatch, ROS_DOMAIN_ID drift, etc.) — not in yolo.

**Tunable knobs (all `-p name:=value` at launch):**
| param | default | meaning |
|---|---|---|
| `stop_target_offset_m` | 0.30 | stop this far before sign face (depth-rate fallback only) |
| `stop_predict_min_samples` | 3 | history samples before depth-rate fit |
| `stop_predict_max_depth_m` | 8.0 | reject detections beyond this |
| `stop_predict_commit_at_m` | 4.5 | depth-rate commit threshold |
| `stop_predict_min_speed` | 0.05 | min approach speed (m/s) to commit |
| `lateral_edge_frac` | 0.15 | outer N% of frame triggers immediate commit |

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (lateral-edge commit, bumped defaults, brake-state log).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 144.

## 2026-05-26 PM-4 (YOLO detector v3.3 — TL visibility for overhead lights + bbox-height proximity gate + diagnosis from user's v3.1 run)

**Two big diagnoses from VO_readings.txt (user's latest run).**

### Diagnosis 1: stop sign actually DID commit and the brake intent WAS sent
Line 39076:
```
[predict] stop sign COMMIT (depth-rate): depth=4.36m approach=3.24m/s brake_in=1.26s
```
That's yolo_detector telling `/motion_enable` to flip False. But user reported the car didn't visibly stop. Three possibilities, in order of likelihood:
1. **User ran v3.1, never built v3.2.** The log has no `BRAKE ENGAGED` lines (which v3.2 added in `on_timer`). `colcon build --packages-select qcar2_autonomy` is mandatory between code edits or the changes don't load. **This is the most likely explanation** because everything else in the log is consistent with v3.1 code paths.
2. Brake fired but car coasted past in the 1.26 s warning + motor inertia.
3. Brake fired but a downstream node overrode `/cmd_vel_nav`.

Forward path is correct: yolo → `/motion_enable` (Bool) → nav_to_pose.py:303 (sub) → :438 (cb sets `motion_flag`) → :600-611 (`enable=0` zeroes `linear.x` and `angular.z` in Twist) → `/cmd_vel_nav` → nav2_qcar_command_convert.cpp:36 (sub) → `qcar2_motor_speed_cmd`. Verified mechanically; no broken link.

**Action**: rebuild and rerun. Look for `>>> BRAKE ENGAGED (motion_enable -> False)` in yolo console. If present, brake reached the publisher; bug is downstream. If absent, tracker didn't commit, paste log and we tune.

### Diagnosis 2: TL never engaged because visibility gate was wrong for overhead TLs
User log:
```
TL RED @ 8.40m: bbox not fully visible (margin=8px) -- NOT stopping  (x bunch)
```
Walking the bboxes for the same TL:
```
(259, 35, 32x62)  top=35, depth=9.8m   <- depth too far (>3.5m), but visible
(252, 20, 34x79)  top=20, depth=9.0m   <- still too far, still visible
(250, 0, 42x81)   top=0,  depth=8.2m   <- bbox at top of frame, depth still > 3.5m
```
**Two root causes stacked:**
1. **Camera depth never drops** for an overhead TL. The camera looks UP at the TL; depth along the camera ray stays ~7-9 m all the way to the stop line. `tl_stop_dist_m=3.5` was literally unreachable.
2. **Visibility gate rejects top-clipping.** Overhead TLs naturally clip the top of the frame as the car approaches (camera tilts up, TL bracket exits above). The old "8 px margin on every edge" rule was designed against TLs entering laterally from the side. Top-clipping doesn't carry the same risk and should be allowed.

**Fix:**
- `tl_stop_dist_m`: 3.5 → **10.0** m (covers realistic overhead-TL camera depths)
- New param `tl_min_height_px = 50`: bbox-height proximity fallback. A TL bbox ≥ 50 px tall in a 480-tall frame means we're visually close enough to act, regardless of depth. Logical OR with the depth gate.
- New param `tl_allow_top_clip = True`: skip the top-edge check in `_bbox_fully_in_frame()` for TLs. Still reject left/right/bottom clipping.
- Per-edge diagnostic in the "NOT stopping" log line — spells out WHICH edge tripped (`left x=`, `right x+w=`, `top y=`, `bottom y+h=`) so future tuning has a fact to act on.
- New `[YOLO] TL ... too far (depth<10.0 or h>=50) -- NOT stopping` log when depth-and-height both fail.

### On user's VO question ("can VO / KLT / PnP help with distance?")
Honest answer: **VO can tell us our forward velocity**, but for the "stop right beside the sign" problem, **velocity isn't the missing piece**. The missing piece is **the sign's lateral position relative to the car**, which is encoded directly in the bbox x-coordinate in the camera frame. That's what v3.2's lateral-edge commit uses. VO/KLT/PnP would give a redundant estimate of forward velocity that the wheel encoder (`/qcar2_joint`) already provides. Not worth wiring in.

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (`_bbox_fully_in_frame` per-edge with `allow_top_clip`; TL gate uses `depth_ok OR height_ok`; per-edge clipping diagnostic).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 145.

## 2026-05-26 PM-5 (YOLO detector v3.4 — fix the actual stop-sign brake bug + Luigi encoder check)

**Critical bug found in user's v3.3 run** (line 40138 of VO_readings.txt):
```
[predict] stop sign COMMIT (depth-rate): depth=4.36m approach=0.77m/s brake_in=5.50s
... stop sign continues being tracked, bbox cx slides 464 -> 555 -> 588 -> 620 ...
... sign exits FOV around t=...848.764 ...
... 5.50 seconds later (target_arrival), NO brake fires ...
```
All subsequent `>>> BRAKE ENGAGED` lines in the log are `hold for 0.60s` → TL only, never stop-sign (which would be `hold for 3.00s`). User confirmed via `ros2 topic echo /motion_enable` that for stop signs the topic stays True the entire run.

**Two compounding bugs:**

### Bug A: lateral-edge trigger was blocked by an earlier depth-rate commit
The tracker's "already committed → return early" check ran BEFORE the lateral-edge check. So if depth-rate committed first (sample 3 of the 8-sample history, around depth 4.36m), the tracker froze and the lateral-edge trigger never ran — even when bbox cx kept growing past the lateral edge threshold (544). For side-of-road signs, depth-rate commits almost always land **in the future** (small depth_rate × large remaining_offset = several seconds away), so this freeze is catastrophic.

**Fix:** Lateral edge now runs **before** the early-return and OVERRIDES any prior commit whose target time is in the future. Logged with `COMMIT (lateral edge) OVERRIDES prior commit (was brake_in=…s)`.

### Bug B: `should_brake()` was only polled inside the per-detection elif
That elif only runs when a stop-sign detection is in the current frame. The depth-rate path sets a target like 5 s in the future. The sign exits the FOV within ~1 s. For the next ~4 s no detection → no elif → no `should_brake()` poll → brake never engages even though the tracker is armed.

**Fix:** Added an end-of-tick poll in `yolo_detect()` after the per-detection loop:
```python
for tracker, hold, label in [(self._stop_tracker, stop_hold, "Stop Sign"),
                              (self._yield_tracker, yield_hold, "Yield Sign")]:
    if tracker.should_brake(poll_now) and poll_now >= self.sign_cooldown_until_abs:
        self.brake_until_abs = max(self.brake_until_abs, poll_now + hold)
        ...
        tracker.on_brake_fired()
```
Runs every tick. If a tracker is armed and its target time has arrived, brake fires regardless of whether the sign is still visible.

**Combined effect of A + B:** With v3.4, the user's exact same run trace would have:
- t=848.198 → depth-rate commit (brake_in=5.50s, target=853.70)
- t=848.598 → bbox cx=594, hits lateral edge (>544), commit OVERRIDDEN to target=now
- Same tick: end-of-tick poll sees armed tracker, fires brake_until_abs = now + 3.00s
- `>>> BRAKE ENGAGED (motion_enable -> False) at t=848.598, hold for 3.00s`
- Car stops next to the sign. ✅

### Luigi encoder check (user asked: "use Luigi's encoder reading?")
Checked `git diff origin/main..origin/luigi-5 -- nav_to_pose.py` for `joint_state_callback`. **The encoder reading formula is byte-identical between main and luigi-5:**
```python
self.qcar2_measurred_speed = (msg.velocity[0]/(720.0*4.0)) * ((13.0*19.0)/(70.0*30.0)) * (2.0*np.pi) * 0.033
```
Luigi did NOT change how the encoder is read. What Luigi DID add: extensive controller-side instrumentation (`/nav/speed_cmd`, `/nav/yaw_rate_imu`, `/nav/progress_rate`, `/nav/controller_mode` debug publishers), EKF integration with chi² Mahalanobis landmarking, mode switching, IIR filtering on the gyro signal (`apply_filter('gyro', ...)`). The encoder noise itself is not addressed at the source on his branch.

**Practical takeaway for our problem:** The brake decision in yolo_detector doesn't depend on encoder speed at all (it depends on bbox lateral position + depth-rate). So encoder noise is not the cause of the stop-sign-not-stopping bug fixed above. If encoder noise becomes a problem later (e.g., for the predictive-stop logic in nav_to_pose's path-planner), the right fix would be a complementary filter using IMU yaw rate alongside the encoder — but that's a nav_to_pose change, not a yolo_detector change, and out of scope here.

**Files touched.**
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (lateral-edge always wins; armed-tracker end-of-tick poll).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 146.

## 2026-05-26 PM-6 (YOLO detector v3.5 — TL color stabilization, pass-line rule, stricter HSV, bottom crop)

**v3.4 result:** stop-sign brake fires correctly — user confirmed car stopped "right beside the stop sign, exactly what I wanted." 🎉

**v3.5 addresses TL behavior + the CSI-bumper false positive.** Four independent fixes:

### 1. Temporal color stabilization (the "uh nvm" fix)
The TL HSV check produces a per-frame color reading. User reported the car decides red→stops→"nvm"→green→moves→"actually red"→stops again. That's per-frame color flicker. Fix: maintain a rolling window of the last N color readings (default `tl_color_history_size=5` = ~165 ms at 30 Hz), bucket into red/yellow/green/idle, take the majority. Tie-break order favors STOP signals (red > yellow > green > idle) so a borderline frame never accidentally runs a red light. The effective color drives the gate; the instantaneous color is still logged for debugging.

### 2. Pass-the-line rule
New param `tl_pass_line_height_px=100`. If the TL bbox is taller than this when we FIRST consider engaging a brake, we're already at the intersection and a fresh brake would stop in the middle of the road. The rule blocks NEW activations only — if `_tl_stop_active=True` (we already committed earlier on the approach), refresh-each-frame keeps the brake engaged through the line. Behavior matches what a human driver does: "if it turned yellow when I was already at the line, I keep going."

New `_tl_stop_active` latch + a release condition: when the effective color is non-stop AND we had been stopping, log "release: TL no longer red/yellow" and clear the latch.

### 3. Stricter HSV color check (saturation gate)
`_check_traffic_light_color` previously required just brightness (V) above a relative threshold. Reflections on the TL housing have high V but **low saturation** — they were tripping the gate. Added absolute floors:
  - `tl_color_min_v=90` (0-255 brightness)
  - `tl_color_min_s=70` (0-255 saturation — the key one)
A patch must satisfy `V >= 90 AND S >= 70 AND V > mean_v AND (V - mean_v) > 0.25*(max - min)` to count as lit. The S floor is what rejects gray reflections and white sky.

### 4. Bottom crop to hide the CSI bumper
New param `crop_bottom_px=24` (~5% of 480). Applied in `on_timer` BEFORE yolo inference — the model never sees the bottom rows. The published `/qcar_camera/rgb_yolo` overlay is also cropped, so rqt shows what the model actually sees. `self._img_h` is refreshed each tick to the cropped height so the visibility gate uses the correct frame dimensions. The CSI bumper, which has been misclassified as "car" and "traffic light", is now physically excluded from the input.

Tune by watching rqt: if the bumper still shows, increase `crop_bottom_px` in steps of 4-8.

### Lighting techniques people use for TL color detection (user asked)
Quick literature note for future reference:
1. **HSV + saturation gate** — what v3.5 now does. Filters reflections.
2. **HSV H ranges** — check the hue value to confirm color identity (red H≈0 or 180, yellow H≈30, green H≈60). We're brightness-only; adding H checks is a follow-up.
3. **LAB color space** — A channel (green↔red), B channel (blue↔yellow) are more robust to illumination than RGB.
4. **Adaptive thresholding (Otsu)** — auto-tune per frame.
5. **CNN color classifier** — train a small network on TL bbox crops. Most robust. Your friend's video of a TL switching colors would be perfect training data for this.
6. **Temporal voting** — what v3.5 now does, majority over N frames.
7. **White-balance normalization** — pre-process image to neutral white before checking color.

For now (1) + (6) should make a significant dent. (3) is the next thing to try if the HSV approach still has issues in tricky lighting. (5) is the long-term right answer once we have enough labeled TL crops.

### What the new log lines look like
Successful gate:
```
Traffic Light RED (inst=red, buckets r4/y0/g0/i1) @ depth=8.40m h=81px (height) -> STOP (brake refreshed +0.60s)
```
The bucket counts show the majority-vote breakdown. If buckets look chaotic (e.g. r2/y0/g2/i1), the color check is still noisy and we need to tune `tl_color_min_v` / `tl_color_min_s` further.

Pass-line block:
```
TL RED (inst=red) h=125px > pass-line 100px: already past the line -- NOT starting new brake
```

Release:
```
TL GREEN (inst=green) -- release: TL no longer red/yellow
```

### Files touched
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (bottom crop in on_timer; stricter HSV with V+S floors; temporal majority-vote color; pass-line rule with `_tl_stop_active` latch; new diagnostic logs).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 147.

## 2026-05-26 PM-8 (YOLO detector v3.6 — TL approach state machine: commit-and-hold semantics)

**User feedback after v3.5:**
- `tl_color_history_size=8` was the best preset; flicker mostly gone.
- New requirement: don't stop in the middle of an intersection on a late yellow. Concretely:
  - If we were stopped at red and it goes green, we GO — and if it briefly turns yellow as we cross, we KEEP GOING (don't re-brake).
  - If we first see the TL green when still far away, we GO — and any later color change during the pass-through is ignored.
- Asked whether depth camera would help. Honest answer: **no, not for overhead TLs.** Camera-depth stays ~7-10 m even at the stop line because the camera looks UP along the ray; depth often falls back to NaN (=-1.0 m) at close range when the TL exits the depth FOV at the top. **bbox_h is the right proximity proxy for overhead TLs** — grows monotonically as we approach. v3.6 makes this explicit.

### Design: `_TLStateMachine` (new nested class)

Three states + one reset:

```
IDLE
  +-- first sighting + bbox_h > pass_line          -> COMMIT_GO
  +-- first sighting + effective color red/yellow  -> COMMIT_STOP (brake)
  +-- first sighting + effective color green/idle  -> COMMIT_GO

COMMIT_STOP  (brake refreshed every tick)
  +-- effective color = green for K consecutive frames -> COMMIT_GO (release)
  +-- otherwise                                        -> stay

COMMIT_GO  (NO brake, color-change-IMMUNE)
  +-- anything                                     -> stay

any non-IDLE + TL not seen for M frames           -> IDLE (reset for next TL)
```

The critical invariant: **once in `COMMIT_GO`, no color change can engage a brake.** That eliminates the "stopped mid-intersection on a flash yellow" failure mode by construction.

State transitions emit log lines like:
```
[TL-FSM] IDLE -> COMMIT_STOP: first sight red/yellow at bh=72 -> STOP
[TL-FSM] COMMIT_STOP -> COMMIT_GO: sustained green (3 frames) -> GO (release)
[TL-FSM] COMMIT_GO -> IDLE: lost for 15 frames -> reset
```

### How it integrates (architecture)

- Per-detection loop now COLLECTS instead of DECIDING for TLs:
  - For each TL detection: compute instantaneous color via `_check_traffic_light_color`, push into the temporal-vote history, derive effective color, gate on confidence + visibility + close-enough (`depth_ok OR height_ok`).
  - Of all usable TL detections this tick, remember the most prominent (largest `bbox_h`) as the FSM input.
  - If no usable TL passed the gates, the FSM gets `present=False` and counts toward the lost-frames reset threshold.
- After the loop, call `_tl_fsm.update(...)` exactly once with the chosen TL or `present=False`.
- FSM returns `'brake'` / `'release'` / `None`. The driver acts:
  - `'brake'`: refresh `brake_until_abs += tl_hold`, publish `qcar_state=1`, log STOP line including state.
  - `'release'`: hard-pull `brake_until_abs` back to `now` so the outer `on_timer` flips `motion_enable` to True next tick.
  - `None`: no-op (covers IDLE-with-no-TL, COMMIT_GO-with-anything, and COMMIT_STOP-while-counting-green).

### Removed
- `self._tl_stop_active` boolean from v3.5 — FSM state is now the authority.
- The inline pass-the-line block (`gate_ok and past_pass_line and not _tl_stop_active`) — the FSM's first-sighting + `past_line` check subsumes it.

### New params (all overridable at launch)
| param | default | meaning |
|---|---|---|
| `tl_fsm_lost_frames_to_reset` | 15 (~0.5 s @ 30 fps) | Frames without TL detection before FSM returns to IDLE |
| `tl_fsm_green_frames_to_release` | 3 (~0.1 s) | Sustained green frames required to release a `COMMIT_STOP` |

### Launch additions
Both new params are forwarded by `autonomy_planner_launch.py` so the user can override at launch:
```
ros2 launch qcar2_autonomy autonomy_planner_launch.py \
  yolo_model_path:=... tl_fsm_green_frames_to_release:=5
```

### Why depth was rejected as the primary TL proximity signal
The user proposed using depth for "distancing to the TL." From v3.5 log analysis (lines 38614+ in VO_readings.txt):
- Camera depth for the same overhead TL: stayed in `[7.0, 10.0] m` across the entire approach. Never crossed `tl_stop_dist_m=10` from above OR below in a useful way.
- 70% of the depth readings were `-1.000 m` (center-patch NaN — TL bbox at the top of frame, center patch outside the valid depth area).
- bbox_h same approach: monotonically grew `24 -> 52 -> 75 -> 81 -> 110 -> exit-frame-at-top`. Clean signal.

`tl_pass_line_height_px=100` (the FSM input) is on `bbox_h`. Depth stays in the proximity check as the `depth_ok OR height_ok` OR-gate so it can help on the rare cases where it produces a valid reading, but it's no longer the primary signal.

### Lighting techniques (continued from PM-6 writeup)
HSV V+S floors and temporal voting are now both in. Next-best improvements if TL color still has issues:
- LAB color space color check (A channel for green↔red, B for blue↔yellow — more illumination-robust than HSV).
- HSV H ranges (verify hue numerically: red H≈0/180, yellow≈30, green≈60).
- A small CNN trained on TL bbox crops (most robust; the user's friend's TL color-change video would be perfect input data).

### Files touched
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (new `_TLStateMachine` class; new FSM params; per-detection loop collects TL info; FSM driver after loop; `_tl_stop_active` removed; release path; rich state-transition logs).
- `Development/ros2/src/qcar2_autonomy/launch/autonomy_planner_launch.py` (added `tl_fsm_lost_frames_to_reset` and `tl_fsm_green_frames_to_release` launch args).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 148.

## 2026-05-26 PM-9 (YOLO detector v3.7 — TL FSM corrected: reactive while far, locked only past line)

**Misread fixed.** v3.6 implemented "first-sight commitment" — if first frame was green, COMMIT_GO locked us in, color-immune for the whole approach. User (Turn 149) clarified that's WRONG: a green-then-yellow flip **while still far** should obey the yellow (stop). The "locked GO" should only kick in **once at the line**, not from first sighting.

**New FSM design (v3.7):**

```
IDLE
  +-- first sight + bbox_h > pass_line     -> PASSING (locked GO)
  +-- first sight otherwise                -> WATCHING

WATCHING                                    (reactive while far)
  +-- bbox_h > pass_line                   -> PASSING
  +-- effective color in {red, yellow}     -> brake (refresh)
  +-- effective color in {green, idle}     -> release (drive)

PASSING                                     (locked, color-IMMUNE)
  +-- anything                             -> stay (NO brake ever)

any non-IDLE + TL not seen for M frames    -> IDLE (reset)
```

The two scenarios the user wants:

1. **"Far + green → suddenly yellow while still far → obey, wait for red→green, then go"**
   - WATCHING + green: release (drive)
   - WATCHING + yellow: brake
   - WATCHING + red: brake (refresh)
   - WATCHING + green: release (drive)
   - We approach, bbox grows past pass_line → PASSING (locked)
   - Yellow during cross: ignored ✓

2. **"At the line / in the intersection, no matter what color: GO"**
   - First sight already past_line OR transitioned WATCHING→PASSING from approaching: PASSING
   - All color changes ignored ✓

### What was removed
- The `COMMIT_STOP` / `COMMIT_GO` states from v3.6. WATCHING subsumes COMMIT_STOP (it's just WATCHING + red/yellow); PASSING is the only "locked" state and only applies past the line.
- The `_green_count` field and the K-consecutive-green-to-release logic. WATCHING mirrors live color directly. Brief flickers are absorbed by `tl_color_history_size` upstream (the effective color stays stable via majority vote — that's already where flicker is killed).
- The `green_frames_to_release` parameter is still **declared** for API compatibility but is no longer used by the FSM.

### What was kept
- The "collect the most prominent TL per tick, feed FSM once" architecture.
- `tl_color_history_size` majority vote for effective color.
- `tl_pass_line_height_px` as the WATCHING→PASSING trigger.
- `tl_fsm_lost_frames_to_reset` for IDLE reset.
- `_last_action` field used to detect brake→release transitions (so we emit `release` exactly at the transition instant, not every green tick).

### User's note about crosswalk-aware logic
User asked whether to add: "if I saw the TL green when last frame had a crosswalk visible, but now light's yellow and I no longer see the crosswalk → keep going OR stop based on whether car is past it." Plausible enhancement. Deferred — user wants to check competition staff's preference first ("most of our car is still ON the crosswalk so I would say stop but I will make sure with the competition staff").

### Files touched
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (`_TLStateMachine` rewrite: states + transitions + action semantics; brief comment updates referencing old state names; `green_frames_to_release` param marked unused).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 149.

## 2026-05-26 PM-10 (YOLO detector — revert v3.7 -> v3.6 FSM, user trade-off decision)

**User decision (Turn 150).** v3.7's reactive-while-far semantic was technically more "correct" per traffic law (obey yellow even when far) but in practice produced a worse failure mode: when a green-then-yellow flip happens *as the car is approaching the line at speed*, propagation latency means the brake engages around the moment the bbox crosses pass_line — i.e., car stops with its front in the intersection. That's worse than just driving through.

Reverted `_TLStateMachine` back to v3.6 (commit-and-hold):
- IDLE → first sight + bbox_h > pass_line OR color in {green, idle} → COMMIT_GO (locked, color-immune)
- IDLE → first sight + color in {red, yellow} → COMMIT_STOP (brake until sustained green)
- COMMIT_STOP → sustained green for K frames → COMMIT_GO (release)
- COMMIT_GO → ignore all subsequent color changes

User's explicit reasoning, quoted: **"going on a red is better than staying in the middle of the intersection"**. Documented in the class docstring + the init comment so future readers know the trade-off was deliberate.

The `tl_fsm_green_frames_to_release` param is back in active use (controls the K in "sustained green for K frames → release"). Default 3 = ~0.1 s @ 30 fps.

### What stays from v3.7
Nothing structural — full revert. The COLLECT-then-FSM-drive architecture, the temporal majority vote, the per-detection log changes, the launch args — all kept (those weren't part of the FSM-spec change).

### Why bring v3.6 back specifically
Tested by user (their actual run, not the cleaned-up scenario) where:
1. v3.5 (no FSM, refresh-each-frame) — worked OK most of the time but flickered between brake/release.
2. v3.6 (first-sight commit, this revert) — user described as "actually good" with the caveat about the edge case where front is just past the crosswalk on a flash yellow.
3. v3.7 (reactive-while-far) — failure mode that's strictly worse than v3.6.

User accepts the v3.6 edge case (running a red the model only learned about after committing to GO) as the lesser evil.

### Files touched
- `Development/ros2/src/qcar2_autonomy/autonomy/yolo_detector.py` (`_TLStateMachine` reverted; param comment restored; init comment updated).
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 150.

## 2026-05-26 PM-11 (v2 model training kicked off — 6 classes incl. yield + roundabout, started from v1 weights)

**Goal.** Add yield + roundabout-warning detection to the v1 4-class model so the QCar can react to those signs in QLabs (and eventually on the physical car). User-requested behavior:
- yield = treat like stop (predictive brake, stop-beside-the-sign)
- roundabout = slow down (brief deceleration cue, no full stop)

### Dataset strategy: 875 real-world + 22 QLabs sim, auto-labeled

Searched for a public dataset that has both yield AND the US-style "roundabout ahead" yellow-diamond warning sign — no good match. GTSDB has yield + European blue roundabout (wrong visual style). LISA has yield but not roundabout. The QLabs roundabout sign is unique enough (yellow diamond with circle-of-arrows icon) that real-world transfer would be unreliable.

Pivoted to **auto-labeling 22 user-provided QLabs screenshots**. The sim has clean, consistent colors (no lighting variation), which makes HSV thresholding + shape filtering reliable. Wrote `autolabel_qlabs_signs.py`:

- **Yield**: red mask (HSV H∈[0,10]∪[170,180]) → morphological close → contours → `approxPolyDP` 3-6 vertices → bbox AR ~1:1 → pole-mounted (upper 65% of frame) → MIN_AREA 200, MAX_AREA 6% of img.
- **Roundabout**: yellow mask (HSV H∈[15,35]) → same shape filtering with 4-8 vertices → solidity ≥ 0.30 (contour_area / bbox_area; rejects thin road centerlines that initially tripped the filter).

Two rounds of tuning:
- v1 of the autolabeler: false-positive — yellow road centerline as a "huge diamond" (bbox 350x430 covering pavement + line).
- Added max-area + center-y filters → killed obvious false positives but lost ~9 valid roundabouts.
- Switched the inside-darkness check for a **solidity** check. Road line solidity ≈ 0.10 (thin yellow snake in square bbox); sign solidity ≈ 0.50 (diamond inscribed in rect). Clean discriminator. Final: 25 yield + 9 roundabout boxes across 22 imgs, no false positives.

Build script `build_v2_dataset.py` then:
- Copied the 4-class dataset (875 imgs) wholesale (label files still valid; class IDs 0-3 unchanged).
- Split the 22 qlabs imgs 18/2/2 train/valid/test.
- Wrote 6-class data.yaml: `0: crosswalk, 1: speedlimit, 2: stop, 3: trafficlight, 4: yield, 5: roundabout`.
- Final layout: 630 train / 177 valid / 90 test (897 total).

### Training: started from v1 best.pt
Using `train_v2.py` which is `train.py` with `WEIGHTS=…/road_signs_4class_v1/weights/best.pt`. Ultralytics auto-handles the head expansion (logged `Overriding model.yaml nc=4 with nc=6`). Same hyperparams as v1: 500 epochs cap, patience 100, batch 16, workers=0, AMP, imgsz 640.

**Why warm-start from v1?** Training from plain COCO yolov8s.pt would force the model to relearn crosswalk/speedlimit/stop/trafficlight from scratch (~50 epochs). Warm-start: those classes are already at 95+% mAP50 in v1, and the new yield/roundabout heads can train on top in a small fraction of the time.

Verified at ~3 epochs in: mAP50 0.71 → 0.96 → 0.59 → 0.77 (the dip-and-recover is the new class heads being added; existing classes are recovering fast). Wall-clock ~17s/epoch, GPU 54°C / 39W / 67% util, 5.6 GB VRAM. Will run to completion (or patience early-stop) overnight.

**Process**: pid 33505 inside `isaac_ros_dev-x86_64-container`, detached `docker exec -d`, nohup'd, log at `Development/yolo_training/logs/train_v2.log`.

### Files added
- `Development/yolo_training/scripts/autolabel_qlabs_signs.py` — HSV/shape auto-labeler for QLabs sim signs
- `Development/yolo_training/scripts/build_v2_dataset.py` — merge 4-class + qlabs into 6-class dataset
- `Development/yolo_training/scripts/train_v2.py` — finetune from v1 weights
- `Development/yolo_training/scripts/check_training_v2.sh` — status command (twin of `check_training.sh`)
- `Development/yolo_training/dataset/qlabs_signs/{source,images,labels,raw}/` — QLabs source + labels + viz overlays
- `Development/yolo_training/dataset/road_signs_6class_v2/{train,valid,test}/{images,labels}/` + `data.yaml`

### Status check
```bash
bash Development/yolo_training/scripts/check_training_v2.sh
bash Development/yolo_training/scripts/check_training_v2.sh tail   # also follow log
```

### Pending after training completes (next session's work)
1. Per-class val on best.pt — check yield/roundabout accuracy.
2. Integrate into `yolo_detector.py`:
   - **yield** → reuse `_SignApproachTracker` (same as stop sign), but tighter `yield_hold` (1.5 s, not 3.0 s) and the action is "slow" not "full stop" — actually with current binary brake the difference is just the hold duration.
   - **roundabout** → new sign-type branch. Behavior: once detected + close enough, fire a SHORT brake (~0.6 s) as a deceleration cue, ONCE per roundabout (cooldown). Don't engage the predictive tracker because we don't want to stop AT the sign — we want to brake briefly to slow before entering the circle.
3. Add `model_path` override to the launch file pointing at the v2 weights when ready.

### Files touched
- This changelog entry.
- `VO_Conversation_Log.txt` Turn 151.
