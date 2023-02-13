#pragma once

#include "AST/AST.h"
#include "AST/GenericArgs.h"
#include "AST/PathIdentSegment.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class PathExprSegment : public Node {
  PathIdentSegment ident;
  std::optional<GenericArgs> generics;

public:
  PathExprSegment(Location loc) : Node(loc), ident(loc) {}

  void addIdentSegment(const PathIdentSegment& _ident) { ident = _ident; }
  void addGenerics(GenericArgs generic) { generics = generic; }

  bool hasGenerics() const { return not generics.has_value(); }

  PathIdentSegment getIdent() const { return ident; }
};

} // namespace rust_compiler::ast
