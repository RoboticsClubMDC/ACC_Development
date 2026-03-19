// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from custom_nitros_message_filter_interfaces:msg/SyncStatus.idl
// generated code does not contain a copyright notice
#include "custom_nitros_message_filter_interfaces/msg/detail/sync_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `messages_present`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__init(custom_nitros_message_filter_interfaces__msg__SyncStatus * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(msg);
    return false;
  }
  // exact_time_match
  // messages_present
  if (!rosidl_runtime_c__boolean__Sequence__init(&msg->messages_present, 0)) {
    custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(msg);
    return false;
  }
  return true;
}

void
custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(custom_nitros_message_filter_interfaces__msg__SyncStatus * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // exact_time_match
  // messages_present
  rosidl_runtime_c__boolean__Sequence__fini(&msg->messages_present);
}

bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__are_equal(const custom_nitros_message_filter_interfaces__msg__SyncStatus * lhs, const custom_nitros_message_filter_interfaces__msg__SyncStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // exact_time_match
  if (lhs->exact_time_match != rhs->exact_time_match) {
    return false;
  }
  // messages_present
  if (!rosidl_runtime_c__boolean__Sequence__are_equal(
      &(lhs->messages_present), &(rhs->messages_present)))
  {
    return false;
  }
  return true;
}

bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__copy(
  const custom_nitros_message_filter_interfaces__msg__SyncStatus * input,
  custom_nitros_message_filter_interfaces__msg__SyncStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // exact_time_match
  output->exact_time_match = input->exact_time_match;
  // messages_present
  if (!rosidl_runtime_c__boolean__Sequence__copy(
      &(input->messages_present), &(output->messages_present)))
  {
    return false;
  }
  return true;
}

custom_nitros_message_filter_interfaces__msg__SyncStatus *
custom_nitros_message_filter_interfaces__msg__SyncStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_nitros_message_filter_interfaces__msg__SyncStatus * msg = (custom_nitros_message_filter_interfaces__msg__SyncStatus *)allocator.allocate(sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus));
  bool success = custom_nitros_message_filter_interfaces__msg__SyncStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
custom_nitros_message_filter_interfaces__msg__SyncStatus__destroy(custom_nitros_message_filter_interfaces__msg__SyncStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__init(custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_nitros_message_filter_interfaces__msg__SyncStatus * data = NULL;

  if (size) {
    data = (custom_nitros_message_filter_interfaces__msg__SyncStatus *)allocator.zero_allocate(size, sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = custom_nitros_message_filter_interfaces__msg__SyncStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__fini(custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence *
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * array = (custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence *)allocator.allocate(sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__destroy(custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__are_equal(const custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * lhs, const custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!custom_nitros_message_filter_interfaces__msg__SyncStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence__copy(
  const custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * input,
  custom_nitros_message_filter_interfaces__msg__SyncStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(custom_nitros_message_filter_interfaces__msg__SyncStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    custom_nitros_message_filter_interfaces__msg__SyncStatus * data =
      (custom_nitros_message_filter_interfaces__msg__SyncStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!custom_nitros_message_filter_interfaces__msg__SyncStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          custom_nitros_message_filter_interfaces__msg__SyncStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!custom_nitros_message_filter_interfaces__msg__SyncStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
