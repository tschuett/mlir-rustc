#pragma once

#include "AST/AST.h"
#include "Lexer/Identifier.h"

#include <string>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class LifetimeOrLabel : public Node {
  Identifier label;

public:
  LifetimeOrLabel(Location loc) : Node(loc) {}

  void setLifeTime(const Identifier &l) { label = l; }
  Identifier getLabel() const { return label; }
};

} // namespace rust_compiler::ast
