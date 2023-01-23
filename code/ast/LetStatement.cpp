#include "AST/LetStatement.h"

namespace rust_compiler::ast {

size_t LetStatement::getTokens() { assert(false); }

void LetStatement::setPattern(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> _pat) {
  pat = _pat;
}

void LetStatement::setType(std::shared_ptr<ast::types::Type> _type) {
  type = _type;
}

void LetStatement::setExpression(std::shared_ptr<ast::Expression> _expr) {
  expr = _expr;
}

} // namespace rust_compiler::ast
