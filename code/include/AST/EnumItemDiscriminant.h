#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

namespace rust_compiler::ast {

class EnumItemDisciminant : public Node {
  std::shared_ptr<Expression> expr;
public:
  EnumItemDisciminant(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

}; // namespace rust_compiler::ast
