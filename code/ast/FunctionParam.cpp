#include "AST/FunctionParam.h"

#include "AST/Patterns/IdentifierPattern.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParam::setName(
    std::shared_ptr<ast::patterns::IdentifierPattern> _name) {
  name = _name;
}

void FunctionParam::setType(std::shared_ptr<ast::types::Type> _type) {
  type = _type;
}

size_t FunctionParam::getTokens() {
  return name->getTokens() + 1 + type->getTokens();
}

std::string FunctionParam::getName() {
  assert(name);
  return name->getIdentifier();
}

} // namespace rust_compiler::ast
