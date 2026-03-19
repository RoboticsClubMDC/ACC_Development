// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__TRAITS_HPP_
#define CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace custom_nitros_message_filter_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const SyncStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: exact_time_match
  {
    out << "exact_time_match: ";
    rosidl_generator_traits::value_to_yaml(msg.exact_time_match, out);
    out << ", ";
  }

  // member: messages_present
  {
    if (msg.messages_present.size() == 0) {
      out << "messages_present: []";
    } else {
      out << "messages_present: [";
      size_t pending_items = msg.messages_present.size();
      for (auto item : msg.messages_present) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SyncStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: exact_time_match
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "exact_time_match: ";
    rosidl_generator_traits::value_to_yaml(msg.exact_time_match, out);
    out << "\n";
  }

  // member: messages_present
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.messages_present.size() == 0) {
      out << "messages_present: []\n";
    } else {
      out << "messages_present:\n";
      for (auto item : msg.messages_present) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SyncStatus & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace custom_nitros_message_filter_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use custom_nitros_message_filter_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_nitros_message_filter_interfaces::msg::SyncStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_nitros_message_filter_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_nitros_message_filter_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_nitros_message_filter_interfaces::msg::SyncStatus & msg)
{
  return custom_nitros_message_filter_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_nitros_message_filter_interfaces::msg::SyncStatus>()
{
  return "custom_nitros_message_filter_interfaces::msg::SyncStatus";
}

template<>
inline const char * name<custom_nitros_message_filter_interfaces::msg::SyncStatus>()
{
  return "custom_nitros_message_filter_interfaces/msg/SyncStatus";
}

template<>
struct has_fixed_size<custom_nitros_message_filter_interfaces::msg::SyncStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<custom_nitros_message_filter_interfaces::msg::SyncStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<custom_nitros_message_filter_interfaces::msg::SyncStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__TRAITS_HPP_
