#pragma once

#include "AST/CallParams.h"
#include "AST/Expression.h"

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

  std::shared_ptr<Expression> getFunction() const;

  void setFunction(std::shared_ptr<Expression> f) { function = f; }
  void setParams(const CallParams &p) { callParameter = p; }
};

} // namespace rust_compiler::ast
