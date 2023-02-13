#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class PathIdentSegmentKind {
  Identifier,
  super,
  self,
  Self,
  crate,
  dollarCrate
};

class PathIdentSegment : public Node {
  PathIdentSegmentKind kind;

public:
  PathIdentSegment(Location loc) : Node(loc) {}

  PathIdentSegmentKind getKind() const { return kind; }

};

} // namespace rust_compiler::ast

// FIXME
