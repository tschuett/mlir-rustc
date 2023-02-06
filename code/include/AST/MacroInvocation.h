#pragma once

#include "AST/MacroItem.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroInvocation : public MacroItem {

public:
  MacroInvocation(Location loc)
      : MacroItem(loc, MacroItemKind::MacroInvocation) {}
};

} // namespace rust_compiler::ast
