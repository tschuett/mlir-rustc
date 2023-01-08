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
};

} // namespace rust_compiler::ast
