#pragma once

#include "AST/FunctionParameter.h"

#include <memory>
#include <vector>
#include <span>

namespace rust_compiler::ast {

class FunctionParameters {
  std::vector<std::shared_ptr<FunctionParameter>> params;

public:
  std::span<std::shared_ptr<FunctionParameter>> getParams() {
    return params;
  }
};

} // namespace rust_compiler::ast
