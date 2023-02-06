#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class StructBase : public Node {
  std::shared_ptr<Expression> base;

public:
  StructBase(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
