#pragma once

#include "AST/ClosureParameters.h"
#include "AST/Expression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class ClosureExpression : public ExpressionWithoutBlock {
  bool move;
  std::optional<ClosureParameters> closureParameters;
  std::shared_ptr<Expression> receiver;

  std::variant<std::shared_ptr<Expression>,
               std::pair<std::shared_ptr<types::TypeNoBounds>, BlockExpression>>
      body;

public:
  ClosureExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ClosureExpression) {}
};

} // namespace rust_compiler::ast
