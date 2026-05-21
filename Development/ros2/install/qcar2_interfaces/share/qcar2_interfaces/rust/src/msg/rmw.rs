#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "qcar2_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__qcar2_interfaces__msg__MotorCommands() -> *const std::ffi::c_void;
}

#[link(name = "qcar2_interfaces__rosidl_generator_c")]
extern "C" {
    fn qcar2_interfaces__msg__MotorCommands__init(msg: *mut MotorCommands) -> bool;
    fn qcar2_interfaces__msg__MotorCommands__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MotorCommands>, size: usize) -> bool;
    fn qcar2_interfaces__msg__MotorCommands__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MotorCommands>);
    fn qcar2_interfaces__msg__MotorCommands__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MotorCommands>, out_seq: *mut rosidl_runtime_rs::Sequence<MotorCommands>) -> bool;
}

// Corresponds to qcar2_interfaces__msg__MotorCommands
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

///  Driving command for QCar2 to directly control the Steering angle and Motor throttle
/// std_msgs/Header header

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MotorCommands {
    /// Names of whether to drive steering or throttle. Must be "steering_angle" or "motor_throttle"
    pub motor_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    /// Values for the "command_names".
    /// The order must be identical to the "command_names".
    /// Units are:
    ///   "rad" for "steering_angle"
    ///   "m/s" for "motor_throttle"
    pub values: rosidl_runtime_rs::Sequence<f64>,

}



impl Default for MotorCommands {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !qcar2_interfaces__msg__MotorCommands__init(&mut msg as *mut _) {
        panic!("Call to qcar2_interfaces__msg__MotorCommands__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MotorCommands {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__MotorCommands__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__MotorCommands__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__MotorCommands__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MotorCommands {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MotorCommands where Self: Sized {
  const TYPE_NAME: &'static str = "qcar2_interfaces/msg/MotorCommands";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__qcar2_interfaces__msg__MotorCommands() }
  }
}


#[link(name = "qcar2_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__qcar2_interfaces__msg__BooleanLeds() -> *const std::ffi::c_void;
}

#[link(name = "qcar2_interfaces__rosidl_generator_c")]
extern "C" {
    fn qcar2_interfaces__msg__BooleanLeds__init(msg: *mut BooleanLeds) -> bool;
    fn qcar2_interfaces__msg__BooleanLeds__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BooleanLeds>, size: usize) -> bool;
    fn qcar2_interfaces__msg__BooleanLeds__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BooleanLeds>);
    fn qcar2_interfaces__msg__BooleanLeds__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BooleanLeds>, out_seq: *mut rosidl_runtime_rs::Sequence<BooleanLeds>) -> bool;
}

// Corresponds to qcar2_interfaces__msg__BooleanLeds
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

///  LED commands for QCar2
/// std_msgs/Header header

#[repr(C)]
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
    pub led_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    /// Values for the "led_names".
    /// The order must be identical to the "led_names".
    /// Units are:
    ///   false or true
    pub values: rosidl_runtime_rs::Sequence<bool>,

}



impl Default for BooleanLeds {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !qcar2_interfaces__msg__BooleanLeds__init(&mut msg as *mut _) {
        panic!("Call to qcar2_interfaces__msg__BooleanLeds__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BooleanLeds {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__BooleanLeds__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__BooleanLeds__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { qcar2_interfaces__msg__BooleanLeds__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BooleanLeds {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BooleanLeds where Self: Sized {
  const TYPE_NAME: &'static str = "qcar2_interfaces/msg/BooleanLeds";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__qcar2_interfaces__msg__BooleanLeds() }
  }
}


