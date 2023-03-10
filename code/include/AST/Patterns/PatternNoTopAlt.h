#pragma once

#include "AST/AST.h"
#include "AST/Patterns/Patterns.h"

namespace rust_compiler::ast::patterns {

enum class PatternNoTopAltKind { PatternWithoutRange, RangePattern };

class PatternNoTopAlt : public Node {

  PatternNoTopAltKind kind;

public:
  PatternNoTopAlt(Location loc, PatternNoTopAltKind kind)
      : Node(loc), kind(kind) {}

  PatternNoTopAltKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast::patterns
