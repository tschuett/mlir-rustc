#include "AST/FunctionParameters.h"

namespace rust_compiler::ast {

void FunctionParameters::addFunctionParam(ast::FunctionParam param) {
  params.push_back(param);
}

size_t FunctionParameters::getTokens() {
  //  size_t count = 0;
  //  for (auto &param : params) {
  //    count += param->getCount();
  //  }
  //
  assert(false);

  return 0;
}

} // namespace rust_compiler::ast
