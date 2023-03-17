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
  Visibility(Location loc) : Node(loc), simplePath(loc) {}

  void setKind(VisibilityKind _kind) { kind = _kind; }

  void setPath(const SimplePath &p) { simplePath = p; }

  VisibilityKind getKind() const { return kind; }
  SimplePath getPath() const { return simplePath; }
};

} // namespace rust_compiler::ast
