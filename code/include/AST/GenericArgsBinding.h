#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeExpression.h"
#include "Lexer/Identifier.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class GenericArgsBinding : public Node {
  Identifier identifier;
  std::shared_ptr<types::TypeExpression> type;

public:
  GenericArgsBinding(Location loc) : Node(loc) {}

  void setIdentifier(const Identifier &i) { identifier = i; }
  void setType(std::shared_ptr<types::TypeExpression> e) { type = e; }
};

} // namespace rust_compiler::ast
