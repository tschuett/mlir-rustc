#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeExpression.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class GenericArgsBinding : public Node {
  std::string identifier;
  std::shared_ptr<types::TypeExpression> type;

public:
  GenericArgsBinding(Location loc) : Node(loc) {}

  void setIdentifier(std::string_view i) { identifier = i; }
  void setType(std::shared_ptr<types::TypeExpression> e) { type = e; }
};

} // namespace rust_compiler::ast
