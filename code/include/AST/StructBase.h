#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class StructBase : public Node {
  std::shared_ptr<Expression> path;

public:
  StructBase(Location loc) : Node(loc) {}

  void setPath(std::shared_ptr<Expression> p) { path = p; }
  std::shared_ptr<Expression> getPath() const { return path; }
};

} // namespace rust_compiler::ast
