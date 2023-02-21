#pragma once

#include "AST/MacroItem.h"
#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroInvocationSemi : public Statement {

public:
  MacroInvocationSemi(Location loc)
    : Statement(loc, StatementKind::MacroInvocationSemi) {}
};

} // namespace rust_compiler::ast
