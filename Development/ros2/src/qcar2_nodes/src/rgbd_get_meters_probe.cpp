#if defined(CV_BRDIGE_HAS_HPP)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include "image_transport/image_transport.hpp"

#include "opencv2/core/mat.hpp"

#include "rclcpp/rclcpp.hpp"

#include "quanser/quanser_messages.h"
#include "quanser/quanser_memory.h"
#include "quanser/quanser_video3d.h"

#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/string.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std::chrono_literals;
using namespace std::placeholders;

class RGBDGetMetersProbe : public rclcpp::Node
{
  public:
    RGBDGetMetersProbe()
    : Node("rgbd_get_meters_probe")
    {
        declare_parameter("device_type", std::string("virtual"));
        declare_parameter("camera_num", std::string("0"));
        declare_parameter("device_num", std::string("@tcpip://localhost:18965"));
        declare_parameter("frame_width_depth", 640);
        declare_parameter("frame_height_depth", 480);
        declare_parameter("frame_rate", 30.0);
        declare_parameter("sample_u", -1);
        declare_parameter("sample_v", -1);
        declare_parameter("log_period_frames", 60);
        declare_parameter("expected_raw_divisor", 15707.0);
        declare_parameter("expected_m_per_unit", 0.1);

        getParameters();
        configureCameraIdentifier();

        RCLCPP_WARN(
            get_logger(),
            "rgbd_get_meters_probe opens the depth stream directly. "
            "For the cleanest test, do not run it at the same time as the "
            "normal rgbd node unless you know the backend supports it.");

        openDepthStream();
        allocateBuffers();
    }

    void ImageTransportSetup()
    {
        image_transport::ImageTransport it(shared_from_this());
        raw_pub_ = it.advertise("camera/depth_image_probe_raw", 1);
        meters_pub_ = it.advertise("camera/depth_image_probe_meters", 1);
        status_pub_ = create_publisher<std_msgs::msg::String>(
            "camera/depth_probe_status", 5);

        const auto frame_period_ms =
            std::max(1, static_cast<int>((1.0 / frame_rate_param_) * 1000.0));
        timer_ = create_wall_timer(
            std::chrono::milliseconds(frame_period_ms),
            std::bind(&RGBDGetMetersProbe::publishProbe, this));
    }

    ~RGBDGetMetersProbe() override
    {
        if (depth_stream_ != nullptr) {
            video3d_stream_close(depth_stream_);
        }
        if (capture_ != nullptr) {
            video3d_stop_streaming(capture_);
            video3d_close(capture_);
        }
        if (buffer_depth_raw_ != nullptr) {
            memory_free(buffer_depth_raw_);
        }
        if (buffer_depth_meters_ != nullptr) {
            memory_free(buffer_depth_meters_);
        }
    }

  private:
    void getParameters()
    {
        device_type_ = get_parameter("device_type").as_string();
        camera_num_ = get_parameter("camera_num").as_string();
        device_num_ = get_parameter("device_num").as_string();
        frame_width_depth_ = get_parameter("frame_width_depth").as_int();
        frame_height_depth_ = get_parameter("frame_height_depth").as_int();
        frame_rate_param_ = get_parameter("frame_rate").as_double();
        sample_u_ = get_parameter("sample_u").as_int();
        sample_v_ = get_parameter("sample_v").as_int();
        log_period_frames_ = std::max(
            1, static_cast<int>(get_parameter("log_period_frames").as_int()));
        expected_raw_divisor_ =
            get_parameter("expected_raw_divisor").as_double();
        expected_m_per_unit_ =
            get_parameter("expected_m_per_unit").as_double();
    }

    void configureCameraIdentifier()
    {
        if (device_type_ == "physical") {
            camera_identifier_ = "0";
        } else if (device_type_ == "virtual") {
            camera_identifier_ = "0@tcpip://localhost:18965";
        } else if (device_type_ == "custom") {
            camera_identifier_ = camera_num_ + device_num_;
        } else {
            throw std::runtime_error(
                "Invalid device_type. Expected virtual/physical/custom.");
        }
    }

    void openDepthStream()
    {
        result_ = video3d_open(camera_identifier_.c_str(), &capture_);
        if (result_ < 0) {
            msg_get_error_messageA(
                nullptr, result_, error_message_, sizeof(error_message_));
            throw std::runtime_error(
                std::string("video3d_open failed: ") + error_message_);
        }

        result_depth_ = video3d_stream_open(
            capture_,
            VIDEO3D_STREAM_DEPTH,
            0,
            frame_rate_param_,
            frame_width_depth_,
            frame_height_depth_,
            IMAGE_FORMAT_ROW_MAJOR_GRAYSCALE,
            IMAGE_DATA_TYPE_UINT16,
            &depth_stream_);
        if (result_depth_ < 0) {
            msg_get_error_messageA(
                nullptr, result_depth_, error_message_, sizeof(error_message_));
            throw std::runtime_error(
                std::string("video3d_stream_open(depth) failed: ") +
                error_message_);
        }

        result_ = video3d_start_streaming(capture_);
        if (result_ < 0) {
            msg_get_error_messageA(
                nullptr, result_, error_message_, sizeof(error_message_));
            throw std::runtime_error(
                std::string("video3d_start_streaming failed: ") +
                error_message_);
        }

        RCLCPP_INFO(
            get_logger(),
            "Depth probe opened on %s at %ux%u @ %.1f Hz",
            camera_identifier_.c_str(),
            frame_width_depth_,
            frame_height_depth_,
            frame_rate_param_);
    }

    void allocateBuffers()
    {
        buffer_depth_raw_ = static_cast<t_uint16 *>(
            memory_allocate(
                frame_width_depth_ * frame_height_depth_ * sizeof(t_uint16)));
        buffer_depth_meters_ = static_cast<t_single *>(
            memory_allocate(
                frame_width_depth_ * frame_height_depth_ * sizeof(t_single)));

        if (buffer_depth_raw_ == nullptr || buffer_depth_meters_ == nullptr) {
            throw std::runtime_error(
                "Failed to allocate depth probe buffers.");
        }
    }

    void publishProbe()
    {
        t_video3d_frame depth_frame;
        result_depth_ = video3d_stream_get_frame(depth_stream_, &depth_frame);
        if (result_depth_ < 0) {
            if (result_depth_ != -QERR_WOULD_BLOCK) {
                msg_get_error_messageA(
                    nullptr, result_depth_, error_message_,
                    ARRAY_LENGTH(error_message_));
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 3000,
                    "Depth probe could not get a frame: %s", error_message_);
            }
            return;
        }

        t_uint64 frame_number = 0;
        t_double frame_timestamp = 0.0;
        video3d_frame_get_number(depth_frame, &frame_number);
        video3d_frame_get_timestamp(depth_frame, &frame_timestamp);

        const t_error result_raw =
            video3d_frame_get_data(depth_frame, buffer_depth_raw_);
        const t_error result_meters =
            video3d_frame_get_meters(depth_frame, buffer_depth_meters_);

        if (result_raw < 0 || result_meters < 0) {
            if (result_raw < 0) {
                msg_get_error_messageA(
                    nullptr, result_raw, error_message_,
                    ARRAY_LENGTH(error_message_));
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 3000,
                    "video3d_frame_get_data failed: %s", error_message_);
            }
            if (result_meters < 0) {
                msg_get_error_messageA(
                    nullptr, result_meters, error_message_,
                    ARRAY_LENGTH(error_message_));
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 3000,
                    "video3d_frame_get_meters failed: %s", error_message_);
            }
            video3d_frame_release(depth_frame);
            return;
        }

        cv::Mat raw_depth(
            frame_height_depth_, frame_width_depth_, CV_16UC1, buffer_depth_raw_);
        cv::Mat meters_depth(
            frame_height_depth_, frame_width_depth_, CV_32FC1,
            buffer_depth_meters_);

        std_msgs::msg::Header header;
        header.stamp = get_clock()->now();
        header.frame_id = "depth_probe";

        auto raw_msg = cv_bridge::CvImage(
            header, sensor_msgs::image_encodings::MONO16, raw_depth).toImageMsg();
        auto meters_msg = cv_bridge::CvImage(
            header, sensor_msgs::image_encodings::TYPE_32FC1, meters_depth)
                              .toImageMsg();

        raw_pub_.publish(*raw_msg);
        meters_pub_.publish(*meters_msg);

        ++published_frames_;
        if (published_frames_ % log_period_frames_ == 0) {
            publishStatus(frame_number, frame_timestamp);
        }

        video3d_frame_release(depth_frame);
    }

    void publishStatus(t_uint64 frame_number, t_double frame_timestamp)
    {
        const int u = clampSample(sample_u_, frame_width_depth_);
        const int v = clampSample(sample_v_, frame_height_depth_);
        const size_t idx =
            static_cast<size_t>(v) * frame_width_depth_ + static_cast<size_t>(u);

        const auto raw_value = static_cast<double>(buffer_depth_raw_[idx]);
        const auto meters_value = static_cast<double>(buffer_depth_meters_[idx]);
        const auto estimated_units =
            (expected_raw_divisor_ > 0.0)
                ? raw_value / expected_raw_divisor_
                : 0.0;
        const auto estimated_meters =
            estimated_units * expected_m_per_unit_;

        double min_valid_m = std::numeric_limits<double>::infinity();
        double max_valid_m = 0.0;
        size_t valid_count = 0;
        const size_t total_pixels =
            static_cast<size_t>(frame_width_depth_) * frame_height_depth_;
        for (size_t i = 0; i < total_pixels; ++i) {
            const double m = static_cast<double>(buffer_depth_meters_[i]);
            if (std::isfinite(m) && m > 0.0) {
                ++valid_count;
                min_valid_m = std::min(min_valid_m, m);
                max_valid_m = std::max(max_valid_m, m);
            }
        }
        if (!std::isfinite(min_valid_m)) {
            min_valid_m = 0.0;
        }

        double ratio = 0.0;
        if (meters_value > 1e-6) {
            ratio = estimated_meters / meters_value;
        }

        std_msgs::msg::String status;
        status.data =
            "frame=" + std::to_string(frame_number) +
            " api_ts=" + std::to_string(frame_timestamp) +
            " sample=(" + std::to_string(u) + "," + std::to_string(v) + ")" +
            " raw=" + std::to_string(static_cast<long long>(buffer_depth_raw_[idx])) +
            " meters_api=" + formatDouble(meters_value, 4) +
            " raw_est_m=" + formatDouble(estimated_meters, 4) +
            " est/api=" + formatDouble(ratio, 4) +
            " valid_px=" + std::to_string(valid_count) +
            " min_m=" + formatDouble(min_valid_m, 4) +
            " max_m=" + formatDouble(max_valid_m, 4);

        status_pub_->publish(status);
        RCLCPP_INFO(get_logger(), "[GET_METERS] %s", status.data.c_str());
    }

    static std::string formatDouble(double value, int decimals)
    {
        std::ostringstream stream;
        stream.setf(std::ios::fixed, std::ios::floatfield);
        stream.precision(decimals);
        stream << value;
        return stream.str();
    }

    static int clampSample(int configured_value, int limit)
    {
        if (configured_value < 0) {
            return limit / 2;
        }
        return std::clamp(configured_value, 0, limit - 1);
    }

    std::string device_type_{"virtual"};
    std::string camera_num_{"0"};
    std::string device_num_{"@tcpip://localhost:18965"};
    std::string camera_identifier_;

    t_uint32 frame_width_depth_{640};
    t_uint32 frame_height_depth_{480};
    t_double frame_rate_param_{30.0};
    int sample_u_{-1};
    int sample_v_{-1};
    int log_period_frames_{60};
    double expected_raw_divisor_{15707.0};
    double expected_m_per_unit_{0.1};
    uint64_t published_frames_{0};

    char error_message_[1024]{};

    t_uint16 *buffer_depth_raw_{nullptr};
    t_single *buffer_depth_meters_{nullptr};

    t_video3d capture_{nullptr};
    t_error result_{0};
    t_error result_depth_{0};
    t_video3d_stream depth_stream_{nullptr};

    image_transport::Publisher raw_pub_;
    image_transport::Publisher meters_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<RGBDGetMetersProbe>();
        node->ImageTransportSetup();

        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Starting rgbd_get_meters_probe...");
        executor.spin();
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("rgbd_get_meters_probe"),
                     "Probe startup failed: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
