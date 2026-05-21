// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from qcar2_interfaces:msg/BooleanLeds.idl
// generated code does not contain a copyright notice

#include "qcar2_interfaces/msg/detail/boolean_leds__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_qcar2_interfaces
const rosidl_type_hash_t *
qcar2_interfaces__msg__BooleanLeds__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x04, 0x04, 0x47, 0xdb, 0xe9, 0x5c, 0x85, 0xa3,
      0x1d, 0x8e, 0xe6, 0x28, 0xfe, 0xc2, 0xba, 0x09,
      0x33, 0xe0, 0x1d, 0x8f, 0x59, 0x0a, 0xd5, 0xea,
      0xd6, 0xfc, 0xca, 0xdc, 0x7a, 0x13, 0xae, 0x5d,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char qcar2_interfaces__msg__BooleanLeds__TYPE_NAME[] = "qcar2_interfaces/msg/BooleanLeds";

// Define type names, field names, and default values
static char qcar2_interfaces__msg__BooleanLeds__FIELD_NAME__led_names[] = "led_names";
static char qcar2_interfaces__msg__BooleanLeds__FIELD_NAME__values[] = "values";

static rosidl_runtime_c__type_description__Field qcar2_interfaces__msg__BooleanLeds__FIELDS[] = {
  {
    {qcar2_interfaces__msg__BooleanLeds__FIELD_NAME__led_names, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {qcar2_interfaces__msg__BooleanLeds__FIELD_NAME__values, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
qcar2_interfaces__msg__BooleanLeds__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {qcar2_interfaces__msg__BooleanLeds__TYPE_NAME, 32, 32},
      {qcar2_interfaces__msg__BooleanLeds__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "## LED commands for QCar2\n"
  "#std_msgs/Header header\n"
  "\n"
  "# Names of the LED.\n"
  "# Must be the following:\n"
  "#    \"left_outside_brake_light\"\n"
  "#    \"left_inside_brake_light\"\n"
  "#    \"right_inside_brake_light\"\n"
  "#    \"right_outside_brake_light\"\n"
  "#    \"left_reverse_light\"\n"
  "#    \"right_reverse_light\"\n"
  "#    \"left_rear_signal\"\n"
  "#    \"right_rear_signal\"\n"
  "#    \"left_outside_headlight\"\n"
  "#    \"left_middle_headlight\"\n"
  "#    \"left_inside_headlight\"\n"
  "#    \"right_inside_headlight\"\n"
  "#    \"right_middle_headlight\"\n"
  "#    \"right_outside_headlight\"\n"
  "#    \"left_front_signal\"\n"
  "#    \"right_front_signal\"\n"
  "string[] led_names\n"
  "\n"
  "# Values for the \"led_names\".\n"
  "# The order must be identical to the \"led_names\".\n"
  "# Units are:\n"
  "#   false or true\n"
  "bool[] values";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
qcar2_interfaces__msg__BooleanLeds__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {qcar2_interfaces__msg__BooleanLeds__TYPE_NAME, 32, 32},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 701, 701},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
qcar2_interfaces__msg__BooleanLeds__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *qcar2_interfaces__msg__BooleanLeds__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
