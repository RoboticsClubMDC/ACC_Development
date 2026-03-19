// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace custom_nitros_message_filter_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void SyncStatus_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_nitros_message_filter_interfaces::msg::SyncStatus(_init);
}

void SyncStatus_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_nitros_message_filter_interfaces::msg::SyncStatus *>(message_memory);
  typed_message->~SyncStatus();
}

size_t size_function__SyncStatus__messages_present(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<bool> *>(untyped_member);
  return member->size();
}

void fetch_function__SyncStatus__messages_present(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & member = *reinterpret_cast<const std::vector<bool> *>(untyped_member);
  auto & value = *reinterpret_cast<bool *>(untyped_value);
  value = member[index];
}

void assign_function__SyncStatus__messages_present(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & member = *reinterpret_cast<std::vector<bool> *>(untyped_member);
  const auto & value = *reinterpret_cast<const bool *>(untyped_value);
  member[index] = value;
}

void resize_function__SyncStatus__messages_present(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<bool> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SyncStatus_message_member_array[3] = {
  {
    "stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces::msg::SyncStatus, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "exact_time_match",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces::msg::SyncStatus, exact_time_match),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "messages_present",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_nitros_message_filter_interfaces::msg::SyncStatus, messages_present),  // bytes offset in struct
    nullptr,  // default value
    size_function__SyncStatus__messages_present,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    fetch_function__SyncStatus__messages_present,  // fetch(index, &value) function pointer
    assign_function__SyncStatus__messages_present,  // assign(index, value) function pointer
    resize_function__SyncStatus__messages_present  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SyncStatus_message_members = {
  "custom_nitros_message_filter_interfaces::msg",  // message namespace
  "SyncStatus",  // message name
  3,  // number of fields
  sizeof(custom_nitros_message_filter_interfaces::msg::SyncStatus),
  SyncStatus_message_member_array,  // message members
  SyncStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  SyncStatus_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SyncStatus_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SyncStatus_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace custom_nitros_message_filter_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<custom_nitros_message_filter_interfaces::msg::SyncStatus>()
{
  return &::custom_nitros_message_filter_interfaces::msg::rosidl_typesupport_introspection_cpp::SyncStatus_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_nitros_message_filter_interfaces, msg, SyncStatus)() {
  return &::custom_nitros_message_filter_interfaces::msg::rosidl_typesupport_introspection_cpp::SyncStatus_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
