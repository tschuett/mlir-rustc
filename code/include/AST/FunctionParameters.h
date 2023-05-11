
#pragma once

#include "AST/FunctionParam.h"
#include "AST/SelfParam.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class FunctionParameters {
  std::vector<FunctionParam> params;
  Location loc;
  std::optional<SelfParam> selfParam;

public:
  FunctionParameters(Location loc) : loc(loc), selfParam(loc) {
    selfParam = std::nullopt;
  }

  std::vector<FunctionParam> getParams() { return params; }

  void addSelfParam(ast::SelfParam selfParam);

  void addFunctionParam(ast::FunctionParam param);

  bool hasSelfParam() const { return selfParam.has_value(); }
  SelfParam getSelfParam() const { return *selfParam; }
};

} // namespace rust_compiler::ast
