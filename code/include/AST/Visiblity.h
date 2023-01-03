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
  Visibility(VisibilityKind kind) : kind(kind) {}

  Visibility(SimplePath simplePath)
      : kind(VisibilityKind::PublicIn), simplePath(simplePath) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
