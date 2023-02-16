#pragma once

#include "AST/FunctionParam.h"
#include "AST/SelfParam.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class FunctionParameters {
  std::vector<FunctionParam> params;
  Location loc;
  ast::SelfParam selfParam;

public:
  FunctionParameters(Location loc) : loc(loc), selfParam(loc) {}

  std::vector<FunctionParam> getParams() { return params; }

  void addSelfParam(ast::SelfParam selfParam);

  void addFunctionParam(ast::FunctionParam param);

  size_t getTokens();
};

} // namespace rust_compiler::ast
