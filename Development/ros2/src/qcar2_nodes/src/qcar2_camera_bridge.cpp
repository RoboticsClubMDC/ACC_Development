// =====================================================================
// qcar2_camera_bridge.cpp
// =====================================================================
//
// STATUS (2026-05-14): NOT THE ACTIVE BRIDGE.
//
// The active single-owner camera bridge is the Python implementation
// at `qcar2_autonomy/autonomy/qcar2_camera_bridge.py` (entry point
// `camera_bridge` in `qcar2_autonomy/setup.py`). The launch files in
// `qcar2_nodes/launch/qcar2_*_launch.py` point at the Python bridge.
//
// Why this C++ implementation is preserved but unused:
//   Over five iterative fixes on 2026-05-14 (path resolver, PIT-matching
//   invocation, -QERR_WOULD_BLOCK handling, switch from `stream_receive`
//   to `stream_receive_byte_array`, switch from `qcomm_*` to `stream_*`
//   with explicit recv_buffer_size=4,915,200, and finally bumping the
//   kernel `net.core.rmem_max` from 212,992 to 16,777,216), this bridge
//   stayed pinned at ~0.3 fps vs the ~30 fps PIT's Python wrapper
//   achieves on the same Jetson. Quanser's library does something
//   inside the Python binding layer that pure-C callers cannot reach
//   from outside — likely an internal ring-buffer pull from the kernel
//   that runs in a thread invisible to the C-level `stream_*` API.
//   Full forensic in `qcar2_autonomy/VO_CHANGELOG.md` under the
//   2026-05-14 entries.
//
// This file stays buildable and the executable still installs under
// `qcar2_nodes/qcar2_camera_bridge` so it remains a reference. If
// future Quanser SDK updates expose the missing piece (e.g. a buffer
// property that actually takes effect on TCP, or a sample C client
// for QCar2DepthAlign), this file is the starting point.
//
// =====================================================================
//
// Single-owner camera bridge for the physical QCar 2.
//
// Owns the RealSense via the Quanser depth-align runtime
// (QCar2DepthAlign.rt-linux_qcar2) and re-publishes the aligned frames
// as standard ROS topics so every other camera consumer (VO, YOLO,
// traffic detector, lane detector, ...) can simply subscribe.
//
// Why this exists
// ---------------
// The legacy rgbd.cpp node owns the RealSense directly via the
// `video3d_*` C API and publishes raw MONO16 depth that is NOT
// registered to the color image. The PIT helper QCar2DepthAligned
// (Python class in pit.YOLO.utils) spawns a Quanser runtime
// (`QCar2DepthAlign.rt-linux_qcar2`) that performs depth-to-color
// alignment at the driver level and streams (depth, RGB) packets
// over TCP. Several components ended up using QCar2DepthAligned
// directly (e.g. yolo_detector.py), creating multiple camera owners
// and hardware contention.
//
// This bridge consolidates the physical camera path: it is the sole
// client of the Quanser depth-align runtime. Subscribers get aligned
// 32FC1 depth and a bgr8 color image on the same topic names
// rgbd.cpp used (/camera/color_image, /camera/depth_image), so the
// existing wiring keeps working.
//
// Packet protocol (matches pit.YOLO.utils.QCar2DepthAligned)
// ----------------------------------------------------------
// The QCar2DepthAlign runtime sends 480 * 640 * 4 float32 elements
// per frame in Fortran (column-major) order. Channels are:
//   channel 0 : depth in meters (single precision)
//   channel 1 : red   (0..255)
//   channel 2 : green (0..255)
//   channel 3 : blue  (0..255)
// Total packet size = 480 * 640 * 4 * 4 bytes = 4,915,200 bytes.
//
// The Fortran layout means the linear offset of element (row=i,
// col=j, channel=k) within the byte buffer is
//   (k * 480 * 640 + j * 480 + i) * sizeof(float)
//
// ROS topics published
// --------------------
//   /camera/color_image  sensor_msgs/Image, encoding "bgr8"
//   /camera/depth_image  sensor_msgs/Image, encoding "32FC1"  (meters)
//
// Parameters
// ----------
//   device_type            "physical" | "virtual"   (default: "physical")
//   auto_start_runtime     bool                     (default: true)
//   runtime_path           string  path to QCar2DepthAlign.rt-linux_qcar2
//                          (default: discovered under MDC_libraries
//                          resources tree)
//   quarc_target_uri       string  (default: "tcpip://localhost:17000")
//   pit_uri                string  (default: "tcpip://localhost:17003")
//   timer_rate_hz          double  publishing-side timer rate
//                          (default: 60.0; receive worker thread runs
//                          continuously, so the timer simply paces ROS
//                          publish based on the freshest buffered frame)
//
// =====================================================================

#if defined(CV_BRDIGE_HAS_HPP)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/header.hpp"

#include "opencv2/core/mat.hpp"

#include "quanser/quanser_communications.h"
#include "quanser/quanser_errors.h"
#include "quanser/quanser_messages.h"
#include "quanser/quanser_stream.h"
#include "quanser/quanser_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {
// Frame geometry: matches QCar2DepthAligned's receiveBuffer shape
constexpr int kH = 480;
constexpr int kW = 640;
constexpr int kChannels = 4;
constexpr int kPacketFloats = kH * kW * kChannels;
constexpr std::size_t kPacketBytes = kPacketFloats * sizeof(float);
}  // namespace


class CameraBridge : public rclcpp::Node {
public:
    CameraBridge()
    : rclcpp::Node("qcar2_camera_bridge"),
      stream_(nullptr),
      conn_open_(false),
      stop_requested_(false),
      have_frame_(false),
      runtime_spawn_ok_(false)
    {
        // ---- Parameters ----
        device_type_       = this->declare_parameter<std::string>("device_type", "physical");
        auto_start_runtime_= this->declare_parameter<bool>("auto_start_runtime", true);
        // Empty default => use the candidate-list resolver in
        // resolveRuntimePath(). The PIT helper resolves this relative to its
        // own __file__, so the binary actually lives in the Quanser install
        // tree at /home/nvidia/Documents/Quanser/0_libraries/... on the QCar2,
        // NOT inside ACC_Development. Leaving the param empty lets the
        // resolver search all known locations.
        runtime_path_      = this->declare_parameter<std::string>(
            "runtime_path", "");
        quarc_target_uri_  = this->declare_parameter<std::string>(
            "quarc_target_uri", "tcpip://localhost:17000");
        pit_uri_           = this->declare_parameter<std::string>(
            "pit_uri", "tcpip://localhost:17003");
        timer_rate_hz_     = this->declare_parameter<double>("timer_rate_hz", 60.0);

        RCLCPP_INFO(this->get_logger(),
                    "qcar2_camera_bridge starting (device_type=%s, pit_uri=%s, "
                    "runtime_path=%s, auto_start=%s)",
                    device_type_.c_str(), pit_uri_.c_str(),
                    runtime_path_.c_str(),
                    auto_start_runtime_ ? "true" : "false");

        // ---- Publishers (standard rclcpp publishers; subscribers using
        //      create_subscription receive these transparently) ----
        rgb_pub_   = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/color_image", 5);
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/depth_image", 5);

        // ---- Start the Quanser depth-align runtime (matches PIT behavior) ----
        if (auto_start_runtime_ && device_type_ == "physical") {
            startRuntime();
            if (!runtime_spawn_ok_) {
                RCLCPP_ERROR(this->get_logger(),
                    "Aborting bridge init: the Quanser depth-align runtime "
                    "did not start. Set parameter 'runtime_path' to the "
                    "correct location and relaunch. Bridge will publish no "
                    "frames in this state.");
                return; // Skip connect + worker spawn.
            }
        } else if (device_type_ == "virtual") {
            RCLCPP_WARN(this->get_logger(),
                "Virtual mode requested. This bridge currently targets the "
                "physical QCar2DepthAlign runtime path. For virtual support, "
                "extend startRuntime() to use the QLabs video3d port + "
                "homography path described in pit.YOLO.utils.QCar2DepthAligned.");
        } else {
            // auto_start_runtime_ == false: user manages the runtime.
            // Mark spawn as OK so we proceed to connect.
            runtime_spawn_ok_ = true;
        }

        // ---- Open the Quanser TCP stream ----
        if (!connectStream()) {
            RCLCPP_ERROR(this->get_logger(),
                "Initial stream connect failed. The worker thread will keep "
                "retrying. Check that the depth-align runtime is up and that "
                "no other client is already attached to %s.", pit_uri_.c_str());
        }

        // ---- Spawn the receive-worker thread ----
        stop_requested_.store(false);
        worker_ = std::thread(&CameraBridge::workerLoop, this);

        // ---- Publish timer ----
        const auto period_ms = std::chrono::milliseconds(
            static_cast<int>(1000.0 / timer_rate_hz_));
        timer_ = this->create_wall_timer(
            period_ms, std::bind(&CameraBridge::onTimer, this));
    }

    ~CameraBridge() override {
        stop_requested_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
        if (conn_open_ && stream_ != nullptr) {
            stream_shutdown(stream_);
            stream_close(stream_);
            stream_ = nullptr;
            conn_open_ = false;
        }
        if (auto_start_runtime_ && device_type_ == "physical"
                && runtime_spawn_ok_) {
            stopRuntime();
        }
    }

private:
    // ---- Resolve the path to QCar2DepthAlign.rt-linux_qcar2 ----
    // If the user supplied `runtime_path` and the file exists there, use it.
    // Otherwise scan a candidate list. Returns true and sets `out` on success.
    //
    // Priority order intentionally favors copies that live inside
    // ACC_Development so the project stays self-contained on a fresh
    // clone. All four copies were verified byte-identical
    // (md5 58572dc0d62e8535140afb45f5eaf554) — order matters only for
    // self-containment / fallback behavior, not correctness.
    bool resolveRuntimePath(std::string& out) {
        std::vector<std::string> candidates;
        if (!runtime_path_.empty()) {
            candidates.push_back(runtime_path_);
        }
        // Allow override via environment variable (placed first below the
        // explicit parameter so users can switch without rebuilding).
        const char* env = std::getenv("QCAR2_DEPTHALIGN_RUNTIME");
        if (env && *env) candidates.push_back(env);

        // In-repo copy: preferred default so the project does not depend
        // on the Quanser system install location.
        candidates.push_back(
            "/home/nvidia/Documents/ACC_Development/docker/0_libraries/resources/"
            "applications/QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2");
        // In-repo backup mirror, used if docker/ is unavailable.
        candidates.push_back(
            "/home/nvidia/Documents/ACC_Development/backup/Quanser_Academic_"
            "Resources/0_libraries/resources/applications/QCarDepthAlign/"
            "QCar2DepthAlign.rt-linux_qcar2");
        // System install location (Quanser canonical). Final fallback for
        // setups where the in-repo copies have been pruned.
        candidates.push_back(
            "/home/nvidia/Documents/Quanser/0_libraries/resources/applications/"
            "QCarDepthAlign/QCar2DepthAlign.rt-linux_qcar2");

        for (const auto& path : candidates) {
            if (path.empty()) continue;
            struct stat st;
            if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                out = path;
                return true;
            }
        }
        RCLCPP_ERROR(this->get_logger(),
            "Could not locate QCar2DepthAlign.rt-linux_qcar2 in any candidate "
            "path. Set parameter 'runtime_path' or env "
            "QCAR2_DEPTHALIGN_RUNTIME to the binary's full path.");
        for (const auto& path : candidates) {
            if (!path.empty()) {
                RCLCPP_ERROR(this->get_logger(), "  tried: %s", path.c_str());
            }
        }
        return false;
    }

    // ---- Quanser runtime lifecycle (mirrors pit.__initDepthAlign / __stopDepthAlign) ----
    void startRuntime() {
        // Resolve the actual path to the QCar2DepthAlign binary. This is
        // critical: without a valid runtime path, quarc_run will fail
        // ("Unable to download model ... The file could not be found"),
        // return non-zero, and no server will listen on pit_uri.
        std::string resolved_path;
        if (!resolveRuntimePath(resolved_path)) {
            runtime_spawn_ok_ = false;
            return;
        }
        runtime_path_ = resolved_path;
        RCLCPP_INFO(this->get_logger(),
                    "Using QCar2DepthAlign runtime at: %s",
                    runtime_path_.c_str());

        // Empirical fix (2026-05-14, Step 2 Test 3): match PIT's
        // pit.YOLO.utils.QCar2DepthAligned invocation EXACTLY. The bridge
        // previously wrapped the runtime path in double quotes and passed a
        // quoted full path to `-q -Q`. The manual test on this Jetson
        // demonstrated that the quoted form fails silently while the
        // unquoted form starts the runtime cleanly on port 17003 (and 18777
        // as a side default). PIT's stop also uses just the basename of the
        // model, not a full path, so we mirror that here.
        const std::string basename = std::filesystem::path(runtime_path_)
                                          .filename().string();

        // 1) Stop any previously-running instance quietly (`-q -Q` =
        //    quiet + quit; takes the model BASENAME, not a path).
        std::string stop_cmd = "quarc_run -t " + quarc_target_uri_ +
                               " -q -Q " + basename;
        std::system(stop_cmd.c_str());

        // 2) Start the runtime. Unquoted path — the runtime_path_ candidate
        //    list guarantees no whitespace.
        std::string start_cmd = "quarc_run -r -t " + quarc_target_uri_ +
                                " " + runtime_path_ +
                                " -uri " + pit_uri_;
        RCLCPP_INFO(this->get_logger(),
                    "Spawning Quanser depth-align runtime: %s",
                    start_cmd.c_str());
        int rc = std::system(start_cmd.c_str());
        if (rc != 0) {
            RCLCPP_ERROR(this->get_logger(),
                "quarc_run returned non-zero (%d). The depth-align runtime "
                "did NOT start. Common causes: binary not at runtime_path, "
                "quarc_run not on PATH, or another runtime is bound to the "
                "URI. Bridge will NOT loop on qcomm_connect — fix the spawn "
                "and restart this node.", rc);
            runtime_spawn_ok_ = false;
            return;
        }
        RCLCPP_INFO(this->get_logger(),
                    "quarc_run returned exit=0; waiting 4s for runtime to "
                    "bind to %s ...", pit_uri_.c_str());
        // PIT sleeps 4 s after spawn before declaring ready. Match that.
        std::this_thread::sleep_for(std::chrono::seconds(4));
        runtime_spawn_ok_ = true;
    }

    void stopRuntime() {
        // `-q -Q` takes the model BASENAME, mirroring PIT's __stopDepthAlign.
        const std::string basename = std::filesystem::path(runtime_path_)
                                          .filename().string();
        std::string cmd = "quarc_run -t " + quarc_target_uri_ +
                          " -q -Q " + basename;
        RCLCPP_INFO(this->get_logger(),
                    "Stopping Quanser depth-align runtime: %s", cmd.c_str());
        std::system(cmd.c_str());
    }

    // ---- Stream connect (uses stream_* C API for explicit buffer sizing) ----
    //
    // Why stream_* instead of qcomm_*: the qcomm_connect() form takes only
    // (uri, non_blocking, &conn) and uses a tiny default receive buffer.
    // With 4.9 MB packets that default forced ~400 receive calls per frame
    // (~12 KB per call) and capped throughput at ~0.3 fps — the heartbeat
    // log in VO_readings.txt -> Step 2 Test 4 (packets=17 over 50 s)
    // confirmed this. PIT's Python wrapper (pal.utilities.stream.BasicStream
    // -> Stream.connect) uses stream_connect with explicit buffer sizes
    // (send=480*640*3, recv=480*640*4*4 = 4,915,200 bytes), which lets each
    // stream_receive return a near-full frame in a single call. We match
    // those exact buffer sizes here.
    //
    // Non-blocking protocol is the same as qcomm_*: stream_connect returns
    // -QERR_WOULD_BLOCK when the TCP handshake is in progress, then
    // stream_poll(stream, t, STREAM_POLL_CONNECT) signals completion.
    // STREAM_POLL_* flags are #define aliases of QCOMM_POLL_* (verified in
    // quanser_stream.h), so no value changes needed.
    bool connectStream() {
        if (conn_open_) return true;

        // PIT's buffer sizes for QCar2DepthAligned (pit.YOLO.utils):
        const t_int send_buffer_size =
            static_cast<t_int>(kW * kH * 3);                  //  921 600 bytes
        const t_int receive_buffer_size =
            static_cast<t_int>(kPacketBytes);                 // 4 915 200 bytes

        const t_error r = stream_connect(pit_uri_.c_str(),
                                         /*non_blocking=*/true,
                                         send_buffer_size,
                                         receive_buffer_size,
                                         &stream_);
        if (r < 0 && r != -QERR_WOULD_BLOCK) {
            char err[256] = {0};
            msg_get_error_messageA(NULL, r, err, sizeof(err));
            RCLCPP_ERROR(this->get_logger(),
                         "stream_connect(%s) failed: %s",
                         pit_uri_.c_str(), err);
            return false;
        }
        // r == 0           -> connection completed synchronously (rare)
        // r == -QERR_WOULD_BLOCK -> in progress; poll for completion
        conn_open_ = true;

        if (r == 0) {
            RCLCPP_INFO(this->get_logger(),
                        "Connected to QCar2DepthAlign stream at %s "
                        "(immediate; send_buf=%ld recv_buf=%ld)",
                        pit_uri_.c_str(),
                        static_cast<long>(send_buffer_size),
                        static_cast<long>(receive_buffer_size));
            return true;
        }

        // Wait for the non-blocking connection to complete.
        t_timeout to = { 0, 5000000, false }; // 5 ms relative
        for (int i = 0; i < 40; ++i) {
            const t_int p = stream_poll(stream_, &to, STREAM_POLL_CONNECT);
            if (p > 0) {
                RCLCPP_INFO(this->get_logger(),
                            "Connected to QCar2DepthAlign stream at %s "
                            "(send_buf=%ld recv_buf=%ld)",
                            pit_uri_.c_str(),
                            static_cast<long>(send_buffer_size),
                            static_cast<long>(receive_buffer_size));
                return true;
            }
            std::this_thread::sleep_for(50ms);
        }
        RCLCPP_WARN(this->get_logger(),
                    "Stream connect did not complete within initial 2 s wait; "
                    "worker will keep polling.");
        return true;  // keep the handle open; later polls may succeed
    }

    // ---- Worker thread: poll, then atomic-receive full packets ----
    //
    // Critical: this uses `stream_receive_byte_array`, NOT plain
    // `stream_receive`. The byte-array variant treats the entire requested
    // byte count as a single atomic unit — it returns 1 only when ALL
    // num_elements bytes have arrived, returns -QERR_WOULD_BLOCK when not
    // enough data is currently buffered, and requires the stream's
    // recv_buffer_size to be >= num_elements (we set it to 4,915,200 in
    // connectStream()). PIT's `pal.utilities.stream.BasicStream.receive`
    // calls `clientStream.receive_byte_array(...)` which is exactly this
    // C function. Using plain `stream_receive` returns partial chunks (~12
    // -23 KB on this Jetson) and capped throughput at ~0.3 fps — see
    // VO_readings.txt -> Step 2 Test 5 heartbeat data.
    void workerLoop() {
        std::vector<float> scratch(kPacketFloats, 0.0f);
        t_byte* const scratch_bytes =
            reinterpret_cast<t_byte*>(scratch.data());
        // Quanser timeout types: seconds, nanoseconds. 10 ms poll budget.
        t_timeout poll_to = { 0, 10000000, false };

        // Heartbeat counters (debug visibility into the receive path).
        std::uint64_t packets_completed = 0;
        std::uint64_t polls_with_data = 0;
        std::uint64_t polls_empty = 0;
        std::uint64_t would_block_count = 0;
        auto last_heartbeat = std::chrono::steady_clock::now();

        while (!stop_requested_.load()) {
            // Periodic heartbeat — every 5 s — so we have visibility into
            // the receive path even when nothing else logs.
            const auto now = std::chrono::steady_clock::now();
            if (now - last_heartbeat >= std::chrono::seconds(5)) {
                RCLCPP_INFO(this->get_logger(),
                    "[bridge worker hb] packets=%lu poll_data=%lu "
                    "poll_empty=%lu would_block=%lu",
                    static_cast<unsigned long>(packets_completed),
                    static_cast<unsigned long>(polls_with_data),
                    static_cast<unsigned long>(polls_empty),
                    static_cast<unsigned long>(would_block_count));
                last_heartbeat = now;
            }

            if (!conn_open_) {
                std::this_thread::sleep_for(250ms);
                connectStream();
                continue;
            }

            const t_int p = stream_poll(stream_, &poll_to, STREAM_POLL_RECEIVE);
            if (p < 0) {
                logQuanserError("stream_poll", p);
                std::this_thread::sleep_for(100ms);
                continue;
            }
            if (p == 0) {
                polls_empty++;
                continue;
            }
            polls_with_data++;

            // Atomic receive of the full 4.9 MB packet. Returns:
            //   1                       on success (entire buffer received)
            //   0                       if connection closed before completion
            //   -QERR_WOULD_BLOCK       if not enough data buffered yet
            //   negative (other)        on error
            const t_int rc = stream_receive_byte_array(
                stream_, scratch_bytes, static_cast<t_uint>(kPacketBytes));

            if (rc == 1) {
                // Full packet — hand off to the publish-side buffer.
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    if (latest_packet_.size() != kPacketFloats) {
                        latest_packet_.assign(kPacketFloats, 0.0f);
                    }
                    std::memcpy(latest_packet_.data(),
                                scratch.data(),
                                kPacketBytes);
                    have_frame_ = true;
                }
                packets_completed++;
                if (packets_completed == 1) {
                    RCLCPP_INFO(this->get_logger(),
                                "First aligned RGBD packet received from "
                                "QCar2DepthAlign runtime; publishing.");
                }
            } else if (rc == 0) {
                RCLCPP_WARN(this->get_logger(),
                            "stream_receive_byte_array returned 0 "
                            "(connection closed gracefully). Will attempt "
                            "to reconnect.");
                conn_open_ = false;
            } else if (rc == -QERR_WOULD_BLOCK) {
                would_block_count++;
                // Not enough buffered yet. Loop back to poll.
            } else {
                logQuanserError("stream_receive_byte_array", rc);
            }
        }
    }

    void logQuanserError(const char* fn, t_int code) {
        char err[256] = {0};
        msg_get_error_messageA(NULL, code, err, sizeof(err));
        RCLCPP_WARN(this->get_logger(),
                    "%s error (%d): %s", fn, static_cast<int>(code), err);
    }

    // ---- Publish timer: build cv::Mats from the latest packet and publish ----
    void onTimer() {
        // Snapshot under the mutex; build/publish outside the lock.
        std::vector<float> snapshot;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (!have_frame_) {
                return; // nothing new to publish
            }
            snapshot = latest_packet_;  // copy
            have_frame_ = false;        // consume; next worker write re-flags
        }

        // Build OpenCV mats from the Fortran-ordered packet.
        // For each pixel (i, j):
        //   depth = snapshot[ 0*kH*kW + j*kH + i ]
        //   R     = snapshot[ 1*kH*kW + j*kH + i ]
        //   G     = snapshot[ 2*kH*kW + j*kH + i ]
        //   B     = snapshot[ 3*kH*kW + j*kH + i ]
        cv::Mat depth_mat(kH, kW, CV_32FC1);
        cv::Mat bgr_mat(kH, kW, CV_8UC3);

        const float* base = snapshot.data();
        const std::size_t cs = static_cast<std::size_t>(kH) * kW;

        for (int j = 0; j < kW; ++j) {
            const std::size_t col_off = static_cast<std::size_t>(j) * kH;
            for (int i = 0; i < kH; ++i) {
                const std::size_t fpos = col_off + i;
                depth_mat.at<float>(i, j) = base[fpos];
                const float r = base[fpos + cs];
                const float g = base[fpos + 2 * cs];
                const float b = base[fpos + 3 * cs];
                bgr_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(b),
                    cv::saturate_cast<uchar>(g),
                    cv::saturate_cast<uchar>(r));
            }
        }

        // Publish RGB
        std_msgs::msg::Header hdr;
        hdr.stamp = this->now();
        hdr.frame_id = "color_image";
        auto rgb_msg = cv_bridge::CvImage(
            hdr, sensor_msgs::image_encodings::BGR8, bgr_mat).toImageMsg();
        rgb_pub_->publish(*rgb_msg);

        // Publish depth (32FC1 meters, aligned to the color grid)
        hdr.frame_id = "depth_image";
        auto depth_msg = cv_bridge::CvImage(
            hdr, sensor_msgs::image_encodings::TYPE_32FC1, depth_mat).toImageMsg();
        depth_pub_->publish(*depth_msg);
    }

    // ---- State ----
    std::string device_type_;
    bool        auto_start_runtime_;
    std::string runtime_path_;
    std::string quarc_target_uri_;
    std::string pit_uri_;
    double      timer_rate_hz_;

    // Opaque Quanser stream handle. We use the `stream_*` C API (in
    // libquanser_communications) rather than `qcomm_*` because only the
    // `stream_*` family lets us size the send/receive buffers explicitly,
    // which is required for 4.9 MB-per-frame packet throughput.
    t_stream stream_;
    bool     conn_open_;

    std::thread       worker_;
    std::atomic<bool> stop_requested_;

    std::mutex         frame_mutex_;
    std::vector<float> latest_packet_;
    bool               have_frame_;

    // True once startRuntime() successfully spawned (or was skipped via
    // auto_start_runtime=false). Used to gate stream connect + cleanup.
    bool runtime_spawn_ok_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
