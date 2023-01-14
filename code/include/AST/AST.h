#pragma once

#include "Location.h"

#include <cstddef>

namespace rust_compiler::ast {

class Node {
protected:
  Location location;

public:
  explicit Node(Location location) : location(location) {}
  virtual ~Node() = default;
  virtual size_t getTokens() = 0;

  rust_compiler::Location getLocation() const { return location; }
};

} // namespace rust_compiler::ast
