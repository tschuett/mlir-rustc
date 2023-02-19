#pragma once

#include "AST/AST.h"
#include "AST/Lifetime.h"
#include "AST/LifetimeBounds.h"

namespace rust_compiler::ast {

class LifetimeParam : public Node {
  Lifetime lifetime;
  ast::LifetimeBounds bounds;

public:
  LifetimeParam(Location loc) : Node(loc), lifetime(loc), bounds(loc) {}

  void setLifetime(Lifetime lt) { lifetime = lt; }

  void setBounds(ast::LifetimeBounds b) { bounds = b; }
};

} // namespace rust_compiler::ast
