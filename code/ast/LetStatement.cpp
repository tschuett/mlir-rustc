#include "AST/LetStatement.h"

#include "AST/VariableDeclaration.h"

namespace rust_compiler::ast {

size_t LetStatement::getTokens() {
  size_t count = 0;

  ++count; // let

  count += pat->getTokens();

  if (type) {
    count += 1 + type->getTokens();
  }

  if (expr)
    count += 1 + expr->getTokens();

  return count + 1;
}

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

std::shared_ptr<ast::patterns::PatternNoTopAlt> LetStatement::getPattern() {
  return pat;
}

std::vector<VariableDeclaration> LetStatement::getVarDecls() {
  std::vector<VariableDeclaration> var;

  std::vector<std::string> lits = pat->getLiterals();
  for (std::string& li: lits) {
  }
}

} // namespace rust_compiler::ast
