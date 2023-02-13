#pragma once

#include "AST/AST.h"

#include <vector>

namespace rust_compiler::ast {

class ArrayElements : public Node {
  // FIXME
public:
  ArrayElements(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
