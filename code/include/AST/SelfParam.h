#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class SelfParam : public Node {

public:
  SelfParam(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
