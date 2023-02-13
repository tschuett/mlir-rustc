#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class CrateRef : public Node {

public:
  CrateRef(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
