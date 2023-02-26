#pragma once

#include "AST/AST.h"
#include "AST/MacroItem.h"

namespace rust_compiler::ast {

enum class MacroFragSpecKind {
  Block,
  Expr,
  Ident,
  Item,
  Lifetime,
  Literal,
  Meta,
  Pat,
  PatParam,
  Path,
  Stmt,
  Tt,
  Ty,
  Vis,
  Unknown
};

class MacroFragSpec : public Node {

  MacroFragSpecKind kind;

public:
  MacroFragSpec(Location loc) : Node(loc) {}

  void setKind(MacroFragSpecKind k) { kind = k; }
};

} // namespace rust_compiler::ast
