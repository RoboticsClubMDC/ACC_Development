#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to qcar2_interfaces__msg__MotorCommands
///  Driving command for QCar2 to directly control the Steering angle and Motor throttle
/// std_msgs/Header header

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MotorCommands {
    /// Names of whether to drive steering or throttle. Must be "steering_angle" or "motor_throttle"
    pub motor_names: Vec<std::string::String>,

    /// Values for the "command_names".
    /// The order must be identical to the "command_names".
    /// Units are:
    ///   "rad" for "steering_angle"
    ///   "m/s" for "motor_throttle"
    pub values: Vec<f64>,

}



impl Default for MotorCommands {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MotorCommands::default())
  }
}

impl rosidl_runtime_rs::Message for MotorCommands {
  type RmwMsg = super::msg::rmw::MotorCommands;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        motor_names: msg.motor_names
          .into_iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        values: msg.values.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        motor_names: msg.motor_names
          .iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        values: msg.values.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      motor_names: msg.motor_names
          .into_iter()
          .map(|elem| elem.to_string())
          .collect(),
      values: msg.values
          .into_iter()
          .collect(),
    }
  }
}


// Corresponds to qcar2_interfaces__msg__BooleanLeds
///  LED commands for QCar2
/// std_msgs/Header header

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BooleanLeds {
    /// Names of the LED.
    /// Must be the following:
    ///    "left_outside_brake_light"
    ///    "left_inside_brake_light"
    ///    "right_inside_brake_light"
    ///    "right_outside_brake_light"
    ///    "left_reverse_light"
    ///    "right_reverse_light"
    ///    "left_rear_signal"
    ///    "right_rear_signal"
    ///    "left_outside_headlight"
    ///    "left_middle_headlight"
    ///    "left_inside_headlight"
    ///    "right_inside_headlight"
    ///    "right_middle_headlight"
    ///    "right_outside_headlight"
    ///    "left_front_signal"
    ///    "right_front_signal"
    pub led_names: Vec<std::string::String>,

    /// Values for the "led_names".
    /// The order must be identical to the "led_names".
    /// Units are:
    ///   false or true
    pub values: Vec<bool>,

}



impl Default for BooleanLeds {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::BooleanLeds::default())
  }
}

impl rosidl_runtime_rs::Message for BooleanLeds {
  type RmwMsg = super::msg::rmw::BooleanLeds;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        led_names: msg.led_names
          .into_iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        values: msg.values.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        led_names: msg.led_names
          .iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        values: msg.values.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      led_names: msg.led_names
          .into_iter()
          .map(|elem| elem.to_string())
          .collect(),
      values: msg.values
          .into_iter()
          .collect(),
    }
  }
}


