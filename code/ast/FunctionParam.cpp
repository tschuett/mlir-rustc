#include "AST/FunctionParam.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParam::setName(std::shared_ptr<ast::patterns::PatternNoTopAlt> _name) {
  name = _name;
}

void FunctionParam::setType(std::shared_ptr<ast::types::Type> _type) {
  type = _type;
}

size_t FunctionParam::getTokens() {
  return name->getTokens() + 1 + type->getTokens();
}

} // namespace rust_compiler::ast
