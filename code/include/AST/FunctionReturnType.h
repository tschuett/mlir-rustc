
#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeExpression.h"

#include <memory>

namespace rust_compiler::ast {

class FunctionReturnType : public Node {
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  FunctionReturnType(Location loc) : Node(loc) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
};

} // namespace rust_compiler::ast
