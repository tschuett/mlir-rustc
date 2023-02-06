#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class FunctionReturnType : public Node {
public:
  FunctionReturnType(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
