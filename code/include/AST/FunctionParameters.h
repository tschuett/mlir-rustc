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

public:
  FunctionParameters(Location loc) : loc(loc) {}

  std::span<FunctionParam> getParams() { return params; }

  void addFunctionParam(ast::FunctionParam param);

  size_t getTokens();
};

} // namespace rust_compiler::ast
