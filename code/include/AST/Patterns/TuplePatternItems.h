#pragma once

#include "AST/Patterns/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class TuplePatternItems : public Node {
public:
  TuplePatternItems(Location loc)
    : Node(loc) {}
};

} // namespace rust_compiler::ast::patterns
