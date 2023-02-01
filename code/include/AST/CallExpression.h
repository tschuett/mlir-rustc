#pragma once

#include "AST/Expression.h"

#include "AST/CallParams.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class CallExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> function;
  std::optional<CallParams> callParameter;
public:
  CallExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::CallExpression) {}
};

} // namespace rust_compiler::ast
