// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__BUILDER_HPP_
#define CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_nitros_message_filter_interfaces
{

namespace msg
{

namespace builder
{

class Init_SyncStatus_messages_present
{
public:
  explicit Init_SyncStatus_messages_present(::custom_nitros_message_filter_interfaces::msg::SyncStatus & msg)
  : msg_(msg)
  {}
  ::custom_nitros_message_filter_interfaces::msg::SyncStatus messages_present(::custom_nitros_message_filter_interfaces::msg::SyncStatus::_messages_present_type arg)
  {
    msg_.messages_present = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_nitros_message_filter_interfaces::msg::SyncStatus msg_;
};

class Init_SyncStatus_exact_time_match
{
public:
  explicit Init_SyncStatus_exact_time_match(::custom_nitros_message_filter_interfaces::msg::SyncStatus & msg)
  : msg_(msg)
  {}
  Init_SyncStatus_messages_present exact_time_match(::custom_nitros_message_filter_interfaces::msg::SyncStatus::_exact_time_match_type arg)
  {
    msg_.exact_time_match = std::move(arg);
    return Init_SyncStatus_messages_present(msg_);
  }

private:
  ::custom_nitros_message_filter_interfaces::msg::SyncStatus msg_;
};

class Init_SyncStatus_stamp
{
public:
  Init_SyncStatus_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SyncStatus_exact_time_match stamp(::custom_nitros_message_filter_interfaces::msg::SyncStatus::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_SyncStatus_exact_time_match(msg_);
  }

private:
  ::custom_nitros_message_filter_interfaces::msg::SyncStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_nitros_message_filter_interfaces::msg::SyncStatus>()
{
  return custom_nitros_message_filter_interfaces::msg::builder::Init_SyncStatus_stamp();
}

}  // namespace custom_nitros_message_filter_interfaces

#endif  // CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__BUILDER_HPP_
