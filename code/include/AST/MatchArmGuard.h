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

  std::shared_ptr<Expression> getGuard() const { return guard; }
};

} // namespace rust_compiler::ast
