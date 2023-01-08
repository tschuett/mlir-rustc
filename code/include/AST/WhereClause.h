#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class WhereClause : public Node {
public:
  explicit WhereClause(Location loc) : Node(loc){};

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
