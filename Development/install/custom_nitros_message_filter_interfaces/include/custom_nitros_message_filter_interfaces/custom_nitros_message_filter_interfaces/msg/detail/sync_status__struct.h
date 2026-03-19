// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_H_
#define CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'messages_present'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/SyncStatus in the package custom_nitros_message_filter_interfaces.
/**
  * Timestamp for this observation
 */
typedef struct custom_nitros_message_filter_interfaces__msg__SyncStatus
{
  builtin_interfaces__msg__Time stamp;
  /// Boolean indicating whether all present messages shared exactly the same Timestamp
  bool exact_time_match;
  /// Boolean array indicating whether or not each message in the synchronization set was present
  rosidl_runtime_c__boolean__Sequence messages_present;
} custom_nitros_message_filter_interfaces__msg__SyncStatus;

// Struct for a sequence of custom_nitros_message_filter_interfaces__msg__SyncStatus.
typedef struct custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence
{
  custom_nitros_message_filter_interfaces__msg__SyncStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_H_
