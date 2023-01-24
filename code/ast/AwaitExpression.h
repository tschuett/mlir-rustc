#oragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class AwaitExpression final : public ExpressionWithoutBlock {

public:
  AwaitExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::AwaitExpression){};
};

} // namespace rust_compiler::ast
