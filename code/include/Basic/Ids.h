#pragma once

#include <cstdint>
#include <limits>

namespace rust_compiler::basic {

/// refers to a Crate
using CrateNum = uint32_t;

/// refers to any node in the current crate
using NodeId = uint32_t;

/// refers to any item in the current crate
using ItemId = uint32_t;

const uint32_t UNKNOWN_CREATENUM = std::numeric_limits<uint32_t>::max();

} // namespace rust_compiler::basic
