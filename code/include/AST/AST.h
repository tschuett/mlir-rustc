#pragma once

#include "Location.h"

#include <cstddef>
#include <vector>

// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/index.html

namespace rust_compiler::ast {

class Node {
  Location location;

public:
  explicit Node(Location location) : location(location) {}
  virtual ~Node() = default;
  virtual size_t getTokens() = 0;

  rust_compiler::Location getLocation() const { return location; }
};

} // namespace rust_compiler::ast
