#pragma once

#include "AST/BlockExpression.h"
#include "AST/ClosureParameters.h"
#include "AST/Expression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class ClosureExpression : public ExpressionWithoutBlock {
  bool move = false;
  std::optional<ClosureParameters> closureParameters;
  std::optional<std::shared_ptr<Expression>> expr;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  ClosureExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ClosureExpression) {}

  bool isMove() const { return move; };
  void setMove() { move = true; }
  void setParameters(const ClosureParameters &cp) { closureParameters = cp; }
  void setBlock(std::shared_ptr<Expression> e) { expr = e; }
  void setExpr(std::shared_ptr<Expression> e) { expr = e; }
  void setType(std::shared_ptr<types::TypeExpression> e) { type = e; }
};

} // namespace rust_compiler::ast
