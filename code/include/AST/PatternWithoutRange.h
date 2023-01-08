#pragma once

#include "AST/AST.h"
#include "AST/PatternNoTopAlt.h"
#include "Location.h"

namespace rust_compiler::ast {

class PatternWithoutRange : public PatternNoTopAlt {

public:
 PatternWithoutRange(Location loc): PatternNoTopAlt(loc) {}
};

} // namespace rust_compiler::ast
