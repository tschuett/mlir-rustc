#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class WhereClause : public Node {
public:
  explicit WhereClause(Location loc) : Node(loc){};
};

} // namespace rust_compiler::ast
