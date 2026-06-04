// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_descriptor_stub.cpp
 * @brief  Weak no-op fallback for register_model_descriptor.
 *
 * When the quick_dot_ai executable is built without libquick_dot_ai_api.so
 * (the legacy main.cpp runner path), model plugin TU constructors still call
 * quick_dot_ai::register_model_descriptor().  This weak definition satisfies
 * the linker; the strong definition in libquick_dot_ai_api.so wins at runtime
 * when the API lib IS loaded.
 */
#include "model_descriptor.h"

namespace quick_dot_ai {
__attribute__((weak)) void
register_model_descriptor(const ModelDescriptor * /*desc*/) {
  // No-op: descriptor registry not available in this build configuration.
}
} // namespace quick_dot_ai
