#pragma once

#include "AST/AST.h"
#include "AST/GenericArgs.h"

#include <string_view>

namespace rust_compiler::ast {

class PathExprSegment : public Node {

public:
  PathExprSegment(Location loc) : Node(loc) {}

  size_t getTokens() override;

  void addIdentSegment(std::string_view ident);
  void addGenerics(GenericArgs generic);
};

} // namespace rust_compiler::ast
