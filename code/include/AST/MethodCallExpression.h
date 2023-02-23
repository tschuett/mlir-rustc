#pragma once

#include "AST/CallParams.h"
#include "AST/Expression.h"
#include "AST/PathExprSegment.h"

namespace rust_compiler::ast {

class MethodCallExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> receiver;
  PathExprSegment method;

  std::optional<CallParams> callParams;

public:
  MethodCallExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::MethodCallExpression),
        method(loc) {}

  PathExprSegment getPath() const;

  void setReceiver(std::shared_ptr<Expression> re) { receiver = re; }
  void setSegment(const PathExprSegment &seg) { method = seg; }
  void setCallParams(const CallParams &pa) { callParams = pa; }
};

} // namespace rust_compiler::ast
