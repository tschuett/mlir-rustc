#pragma once

#include "AST/AST.h"
#include "AST/GenericArgs.h"

#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class PathExprSegment : public Node {
  std::string ident;
  std::vector<GenericArgs> generics;

public:
  PathExprSegment(Location loc) : Node(loc) {}

  size_t getTokens() override;

  void addIdentSegment(std::string_view _ident) { ident = _ident; }
  void addGenerics(GenericArgs generic) { generics.push_back(generic); }
};

} // namespace rust_compiler::ast
