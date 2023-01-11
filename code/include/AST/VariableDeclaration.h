#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class VariableDeclarationKind {};

class VariableDeclaration : Node {
  VariableDeclarationKind kind;

public:
  VariableDeclaration(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast


// FIXME scope path?
