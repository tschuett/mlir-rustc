#include "AST/FunctionParameters.h"

namespace rust_compiler::ast {

void FunctionParameters::addFunctionParam(ast::FunctionParam param) {
  params.push_back(param);
}

} // namespace rust_compiler::ast
