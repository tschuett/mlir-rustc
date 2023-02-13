#include "AST/LetStatementParam.h"

namespace rust_compiler::ast {

void LetStatementParam::setName(std::string _name) {
  name = std::make_shared<std::string>(_name);
}

std::string LetStatementParam::getName() { return *name; }


} // namespace rust_compiler::ast
