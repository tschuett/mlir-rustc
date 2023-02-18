#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class ConstParam : public Node {
public:
  ConstParam(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
