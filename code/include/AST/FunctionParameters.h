#pragma once

#include "AST/FunctionParam.h"
#include "AST/SelfParam.h"

#include <memory>
#include <vector>
#include <span>

namespace rust_compiler::ast {

class FunctionParameters {
  std::vector<std::shared_ptr<FunctionParam>> params;

public:
  std::span<std::shared_ptr<FunctionParam>> getParams() {
    return params;
  }

  size_t getTokens();
};

} // namespace rust_compiler::ast
