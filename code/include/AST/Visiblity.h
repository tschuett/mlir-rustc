#pragma once

#include "AST/AST.h"
#include "AST/SimplePath.h"

namespace rust_compiler::ast {

enum class VisibilityKind {
  Private,
  Public,
  PublicCrate,
  PublicSelf,
  PublicSuper,
  PublicIn
};

class Visibility : public Node {
  VisibilityKind kind;

  SimplePath simplePath;

public:
  Visibility(Location loc, VisibilityKind kind)
      : Node(loc), kind(kind), simplePath(loc) {}

  Visibility(Location loc, SimplePath simplePath)
      : Node(loc), kind(VisibilityKind::PublicIn), simplePath(simplePath) {}
};

} // namespace rust_compiler::ast
