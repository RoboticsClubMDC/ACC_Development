// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_HPP_
#define CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_nitros_message_filter_interfaces__msg__SyncStatus __attribute__((deprecated))
#else
# define DEPRECATED__custom_nitros_message_filter_interfaces__msg__SyncStatus __declspec(deprecated)
#endif

namespace custom_nitros_message_filter_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SyncStatus_
{
  using Type = SyncStatus_<ContainerAllocator>;

  explicit SyncStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->exact_time_match = false;
    }
  }

  explicit SyncStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->exact_time_match = false;
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _exact_time_match_type =
    bool;
  _exact_time_match_type exact_time_match;
  using _messages_present_type =
    std::vector<bool, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<bool>>;
  _messages_present_type messages_present;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__exact_time_match(
    const bool & _arg)
  {
    this->exact_time_match = _arg;
    return *this;
  }
  Type & set__messages_present(
    const std::vector<bool, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<bool>> & _arg)
  {
    this->messages_present = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_nitros_message_filter_interfaces__msg__SyncStatus
    std::shared_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_nitros_message_filter_interfaces__msg__SyncStatus
    std::shared_ptr<custom_nitros_message_filter_interfaces::msg::SyncStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SyncStatus_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->exact_time_match != other.exact_time_match) {
      return false;
    }
    if (this->messages_present != other.messages_present) {
      return false;
    }
    return true;
  }
  bool operator!=(const SyncStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SyncStatus_

// alias to use template instance with default allocator
using SyncStatus =
  custom_nitros_message_filter_interfaces::msg::SyncStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_nitros_message_filter_interfaces

#endif  // CUSTOM_NITROS_MESSAGE_FILTER_INTERFACES__MSG__DETAIL__SYNC_STATUS__STRUCT_HPP_
