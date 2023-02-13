#include "AST/IfLetExpression.h"

namespace rust_compiler::ast {

void IfLetExpression::setPattern(
    std::shared_ptr<ast::patterns::Pattern> _pattern) {
  pattern = _pattern;
}

void IfLetExpression::setScrutinee(std::shared_ptr<ast::Scrutinee> _scrutinee) {
  scrutinee = _scrutinee;
}

void IfLetExpression::setBlock(std::shared_ptr<ast::Expression> _block) {
  block = _block;
}

void IfLetExpression::setTrailing(std::shared_ptr<ast::Expression> _trailing) {
  trailing = _trailing;
}

bool IfLetExpression::containsBreakExpression() {
  if (block->containsBreakExpression())
    return true;
  if (trailing)
    return trailing->containsBreakExpression();
}


} // namespace rust_compiler::ast
