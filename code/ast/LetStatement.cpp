#include "AST/LetStatement.h"

#include "AST/LetStatementParam.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VariableDeclaration.h"

namespace rust_compiler::ast {

void LetStatement::setPattern(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> _pat) {
  pat = _pat;
}

void LetStatement::setType(std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

void LetStatement::setExpression(std::shared_ptr<ast::Expression> _expr) {
  expr = _expr;
}

std::shared_ptr<ast::patterns::PatternNoTopAlt> LetStatement::getPattern() {
  return pat;
}

} // namespace rust_compiler::ast
