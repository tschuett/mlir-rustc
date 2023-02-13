#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class TupleElements : public Node {
  std::vector<std::shared_ptr<Expression>> elements;
  bool trailingComma;

public:
  TupleElements(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
