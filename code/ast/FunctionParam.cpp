#include "AST/FunctionParam.h"

#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Types/TypeExpression.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParam::setName(
    std::shared_ptr<ast::patterns::IdentifierPattern> _name) {
  name = _name;
}

void FunctionParam::setType(std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

std::string FunctionParam::getName() {
  assert(name);
  return name->getIdentifier();
}

void FunctionParam::setAttributes(std::span<OuterAttribute> attr) {
  outerAttributes = {attr.begin(), attr.end()};
}

} // namespace rust_compiler::ast
