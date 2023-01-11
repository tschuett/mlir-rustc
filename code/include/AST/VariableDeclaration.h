#pragma once

#include "AST/AST.h"
#include "AST/Types/Types.h"

namespace rust_compiler::ast {

enum class VariableDeclarationKind {};

class VariableDeclaration : Node {
  VariableDeclarationKind kind;

public:
  VariableDeclaration(Location loc) : Node(loc) {}

  VariableDeclarationKind getKind() const { return kind; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast

// FIXME scope path?
