#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class PatternNoTopAlt : public Node {

public:
  PatternNoTopAlt(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
