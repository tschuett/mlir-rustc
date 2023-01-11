#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class PatternNoTopAltKind { PatternWithoutRange, RangePattern };

class PatternNoTopAlt : public Node {

  PatternNoTopAltKind kind;

public:
  PatternNoTopAlt(Location loc, PatternNoTopAltKind kind)
      : Node(loc), kind(kind) {}

  PatternNoTopAltKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
