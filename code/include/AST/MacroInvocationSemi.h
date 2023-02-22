#pragma once

#include "AST/MacroItem.h"
#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroInvocationSemi : public MacroItem {

public:
  MacroInvocationSemi(Location loc)
      : MacroItem(loc, MacroItemKind::MacroInvocationSemi) {}
};

} // namespace rust_compiler::ast
