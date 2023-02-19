
#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class Lifetime : public Node {
public:
  Lifetime(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
