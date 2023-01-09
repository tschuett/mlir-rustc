#include "AST/FunctionParam.h"

#include <cassert>

namespace rust_compiler::ast {

void FunctionParam::setName(std::shared_ptr<ast::PatternNoTopAlt> _name) {
  name = _name;
}

void FunctionParam::setType(std::shared_ptr<ast::types::Type> _type) {
  type = _type;
}

size_t FunctionParam::getTokens() {
  assert(false);

  return 0;
}

} // namespace rust_compiler::ast

