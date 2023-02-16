#include "AST/BreakExpression.h"

namespace rust_compiler::ast {

void BreakExpression::setExpression(std::shared_ptr<Expression> _break) {
  expr = _break;
}

} // namespace rust_compiler::ast
