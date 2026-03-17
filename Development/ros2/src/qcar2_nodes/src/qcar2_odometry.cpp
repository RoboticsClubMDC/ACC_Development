#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "qcar2_interfaces/msg/motor_commands.hpp"

using namespace std::chrono_literals;

// QCar2 drivetrain constants (from qcar2_hardware.cpp speed_controller)
static constexpr double ENCODER_TICKS_PER_REV = 720.0 * 4.0;   // quadrature encoder
static constexpr double GEAR_RATIO             = (13.0 * 19.0) / (70.0 * 30.0);
static constexpr double WHEEL_RADIUS           = 0.033;          // metres

class QCar2Odometry : public rclcpp::Node
{
public:
    QCar2Odometry()
    : Node("qcar2_odometry"), x_(0.0), y_(0.0), theta_(0.0),
      current_speed_(0.0), current_steering_(0.0)
    {
        this->declare_parameter("wheelbase", 0.256);
        wheelbase_ = this->get_parameter("wheelbase").as_double();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);

        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "qcar2_joint", 1,
            std::bind(&QCar2Odometry::joint_callback, this, std::placeholders::_1));

        motor_cmd_sub_ = this->create_subscription<qcar2_interfaces::msg::MotorCommands>(
            "qcar2_motor_speed_cmd", 1,
            std::bind(&QCar2Odometry::motor_cmd_callback, this, std::placeholders::_1));

        last_time_ = this->get_clock()->now();
    }

private:
    void motor_cmd_callback(const qcar2_interfaces::msg::MotorCommands & msg)
    {
        for (size_t i = 0; i < msg.motor_names.size(); ++i) {
            if (msg.motor_names[i] == "steering_angle") {
                current_steering_ = msg.values[i];
            }
        }
    }

    void joint_callback(const sensor_msgs::msg::JointState & msg)
    {
        if (msg.velocity.empty()) {
            return;
        }

        rclcpp::Time now = msg.header.stamp;
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        // Guard against bad dt at startup or stale messages
        if (dt <= 0.0 || dt > 1.0) {
            return;
        }

        // Convert raw motor speed (ticks/s) to wheel linear speed (m/s)
        // Same formula used in qcar2_hardware.cpp speed_controller
        double motor_ticks_per_sec = msg.velocity[0];
        current_speed_ = (motor_ticks_per_sec / ENCODER_TICKS_PER_REV)
                         * GEAR_RATIO
                         * (2.0 * M_PI)
                         * WHEEL_RADIUS;

        // Ackermann steering: omega = v * tan(delta) / L
        double omega = current_speed_ * std::tan(current_steering_) / wheelbase_;

        // Integrate pose
        theta_ += omega * dt;
        x_     += current_speed_ * std::cos(theta_) * dt;
        y_     += current_speed_ * std::sin(theta_) * dt;

        publish_odom(now, current_speed_, omega);
    }

    void publish_odom(const rclcpp::Time & stamp, double v, double omega)
    {
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, theta_);

        // Publish TF: odom -> base_link
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp    = stamp;
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id  = "base_link";
        tf_msg.transform.translation.x = x_;
        tf_msg.transform.translation.y = y_;
        tf_msg.transform.translation.z = 0.0;
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(tf_msg);

        // Publish Odometry message
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id  = "base_link";

        odom.pose.pose.position.x  = x_;
        odom.pose.pose.position.y  = y_;
        odom.pose.pose.position.z  = 0.0;
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        odom.twist.twist.linear.x  = v;
        odom.twist.twist.angular.z = omega;

        odom_publisher_->publish(odom);
    }

    // State
    double x_, y_, theta_;
    double current_speed_;
    double current_steering_;
    double wheelbase_;
    rclcpp::Time last_time_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<qcar2_interfaces::msg::MotorCommands>::SharedPtr motor_cmd_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QCar2Odometry>());
    rclcpp::shutdown();
    return 0;
}
