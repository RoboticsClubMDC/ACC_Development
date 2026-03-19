// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__rosidl_typesupport_introspection_c.h"
#include "custom_nitros_message_filter_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__functions.h"
#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `messages_present`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  custom_nitros_message_filter_interfaces__msg__SyncStatus__init(message_memory);
}

void custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_fini_function(void * message_memory)
{
  custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(message_memory);
}

size_t custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__size_function__SyncStatus__messages_present(
  const void * untyped_member)
{
  const rosidl_runtime_c__boolean__Sequence * member =
    (const rosidl_runtime_c__boolean__Sequence *)(untyped_member);
  return member->size;
}

const void * custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_const_function__SyncStatus__messages_present(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__boolean__Sequence * member =
    (const rosidl_runtime_c__boolean__Sequence *)(untyped_member);
  return &member->data[index];
}

void * custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_function__SyncStatus__messages_present(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__boolean__Sequence * member =
    (rosidl_runtime_c__boolean__Sequence *)(untyped_member);
  return &member->data[index];
}

void custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__fetch_function__SyncStatus__messages_present(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_const_function__SyncStatus__messages_present(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__assign_function__SyncStatus__messages_present(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_function__SyncStatus__messages_present(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

bool custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__resize_function__SyncStatus__messages_present(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__boolean__Sequence * member =
    (rosidl_runtime_c__boolean__Sequence *)(untyped_member);
  rosidl_runtime_c__boolean__Sequence__fini(member);
  return rosidl_runtime_c__boolean__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_member_array[3] = {
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces__msg__SyncStatus, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "exact_time_match",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces__msg__SyncStatus, exact_time_match),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "messages_present",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces__msg__SyncStatus, messages_present),  // bytes offset in struct
    NULL,  // default value
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__size_function__SyncStatus__messages_present,  // size() function pointer
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_const_function__SyncStatus__messages_present,  // get_const(index) function pointer
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__get_function__SyncStatus__messages_present,  // get(index) function pointer
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__fetch_function__SyncStatus__messages_present,  // fetch(index, &value) function pointer
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__assign_function__SyncStatus__messages_present,  // assign(index, value) function pointer
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__resize_function__SyncStatus__messages_present  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_members = {
  "custom_nitros_message_filter_interfaces__msg",  // message namespace
  "SyncStatus",  // message name
  3,  // number of fields
  sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus),
  custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_member_array,  // message members
  custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_type_support_handle = {
  0,
  &custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_custom_nitros_message_filter_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_nitros_message_filter_interfaces, msg, SyncStatus)() {
  custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_type_support_handle.typesupport_identifier) {
    custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &custom_nitros_message_filter_interfaces__msg__SyncStatus__rosidl_typesupport_introspection_c__SyncStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
