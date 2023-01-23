#pragma once

#include "AST/AST.h"
#include "AST/LetStatement.h"
#include "AST/Types/Types.h"

namespace rust_compiler::ast {

enum class VariableDeclarationKind { FunctionParameter, LetStatement };

class VariableDeclaration : public Node {
  VariableDeclarationKind kind;

public:
  VariableDeclaration(Location loc, VariableDeclarationKind kind)
      : Node(loc), kind(kind) {}

  VariableDeclarationKind getKind() const { return kind; }

  // size_t getTokens() override;

  virtual std::string getName() = 0;
};

} // namespace rust_compiler::ast

// FIXME scope path?
