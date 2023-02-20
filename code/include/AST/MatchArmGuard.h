#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class MatchArmGuard : public Node {
  std::shared_ptr<Expression> guard;

public:
  MatchArmGuard(Location loc) : Node(loc){};

  void setGuard(std::shared_ptr<Expression> g) { guard = g; }
};

} // namespace rust_compiler::ast
