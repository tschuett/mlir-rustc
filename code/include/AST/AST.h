#pragma once

#include <cstddef>

namespace rust_compiler::ast {

class Node {
public:
  virtual ~Node() = default;
  virtual size_t getTokens() = 0;
};

} // namespace rust_compiler::ast
