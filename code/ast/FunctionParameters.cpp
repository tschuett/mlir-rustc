#include "AST/FunctionParameters.h"

namespace rust_compiler::ast {

void FunctionParameters::addFunctionParam(ast::FunctionParam param) {
  params.push_back(param);
}

void FunctionParameters::addSelfParam(ast::SelfParam _self) {
  selfParam = _self;
}

} // namespace rust_compiler::ast
