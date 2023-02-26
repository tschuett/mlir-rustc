#pragma once

#include "AST/AST.h"
#include "AST/MacroFragSpec.h"
#include "AST/MacroMatcher.h"
#include "AST/MacroRepSep.h"
#include "AST/MacroRepOp.h"

namespace rust_compiler::ast {

class MacroMatch : public Node {

public:
  MacroMatch(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
