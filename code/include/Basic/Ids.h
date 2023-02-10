#pragma once

#include <cstdint>

namespace rust_compiler::basic {

// refers to any node in the current crate
using NodeId = uint32_t;

// refers to any item in the current crate
using ItemId = uint32_t;

} // namespace rust_compiler::basic
