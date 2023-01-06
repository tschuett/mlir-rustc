#pragma once

#include "AST/Types/Types.h"

#include <memory>

namespace rust_compiler::ast {

class FunctionParameter {
  std::shared_ptr<ast::types::Type> type;

public:
  std::shared_ptr<ast::types::Type> getType() const { return type; }
};

} // namespace rust_compiler::ast
