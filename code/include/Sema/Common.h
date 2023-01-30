#pragma once

#include <cstdint>

namespace rust_compiler::sema {

// refers to any node in the HIR for the current crate
using AstId = uint32_t;

} // namespace rust_compiler::sema
