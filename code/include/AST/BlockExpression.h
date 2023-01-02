#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Statement.h"

#include <mlir/IR/Location.h>
#include <span>

namespace rust_compiler::ast {

class BlockExpression : public ExpressionWithBlock {
  mlir::Location location;

  std::vector<std::shared_ptr<Statement>> stmts;

public:
  std::span<std::shared_ptr<Statement>> getExpressions();
};

} // namespace rust_compiler::ast
