// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from qcar2_interfaces:msg/MotorCommands.idl
// generated code does not contain a copyright notice

#include "qcar2_interfaces/msg/detail/motor_commands__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_qcar2_interfaces
const rosidl_type_hash_t *
qcar2_interfaces__msg__MotorCommands__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x88, 0xb4, 0x43, 0x1b, 0x1d, 0xb6, 0x80, 0x47,
      0xa2, 0x38, 0x1e, 0xaf, 0xe8, 0x5a, 0x44, 0x09,
      0xae, 0x5d, 0x64, 0xe7, 0x73, 0xe6, 0xca, 0xbf,
      0x9b, 0x7e, 0x1c, 0xd6, 0x6b, 0xb5, 0xc6, 0xee,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char qcar2_interfaces__msg__MotorCommands__TYPE_NAME[] = "qcar2_interfaces/msg/MotorCommands";

// Define type names, field names, and default values
static char qcar2_interfaces__msg__MotorCommands__FIELD_NAME__motor_names[] = "motor_names";
static char qcar2_interfaces__msg__MotorCommands__FIELD_NAME__values[] = "values";

static rosidl_runtime_c__type_description__Field qcar2_interfaces__msg__MotorCommands__FIELDS[] = {
  {
    {qcar2_interfaces__msg__MotorCommands__FIELD_NAME__motor_names, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {qcar2_interfaces__msg__MotorCommands__FIELD_NAME__values, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
qcar2_interfaces__msg__MotorCommands__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {qcar2_interfaces__msg__MotorCommands__TYPE_NAME, 34, 34},
      {qcar2_interfaces__msg__MotorCommands__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "## Driving command for QCar2 to directly control the Steering angle and Motor throttle\n"
  "#std_msgs/Header header\n"
  "\n"
  "# Names of whether to drive steering or throttle. Must be \"steering_angle\" or \"motor_throttle\"\n"
  "string[] motor_names\n"
  "\n"
  "# Values for the \"command_names\".\n"
  "# The order must be identical to the \"command_names\".\n"
  "# Units are:\n"
  "#   \"rad\" for \"steering_angle\"\n"
  "#   \"m/s\" for \"motor_throttle\" \n"
  "float64[] values";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
qcar2_interfaces__msg__MotorCommands__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {qcar2_interfaces__msg__MotorCommands__TYPE_NAME, 34, 34},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 410, 410},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
qcar2_interfaces__msg__MotorCommands__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *qcar2_interfaces__msg__MotorCommands__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
