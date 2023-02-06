#pragma once

#include "AST/AST.h"

#include <vector>

namespace rust_compiler::ast {

class ArrayElements : public Node {
  // FIXME
public:
  ArrayElements(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
