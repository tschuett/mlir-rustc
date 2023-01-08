#include "AST/FunctionParameter.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParameter::setName(std::shared_ptr<ast::PatternNoTopAlt> _name) {
  name = _name;
}

void FunctionParameter::setType(std::shared_ptr<ast::types::Type> _type) {
  type = _type;
}

size_t FunctionParameter::getTokens() {
  assert(false);

  return 0;
}

} // namespace rust_compiler::ast

