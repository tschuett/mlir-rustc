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
  std::shared_ptr<Expression> expr;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  ClosureExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ClosureExpression) {}

  bool isMove() const { return move; };
  void setMove() { move = true; }
  void setParameters(const ClosureParameters &cp) { closureParameters = cp; }
  void setBlock(std::shared_ptr<Expression> e) { expr = e; }
  //  void setExpr(std::shared_ptr<Expression> e) { expr = e; }
  void setType(std::shared_ptr<types::TypeExpression> e) { type = e; }

  std::shared_ptr<Expression> getBody() const { return expr; }

  bool hasParameters() const { return closureParameters.has_value(); }
  ClosureParameters getParameters() const { return *closureParameters; }

  bool hasReturnType() const { return type.has_value(); }
  std::shared_ptr<types::TypeExpression> getReturnType() const { return *type; }
};

} // namespace rust_compiler::ast
