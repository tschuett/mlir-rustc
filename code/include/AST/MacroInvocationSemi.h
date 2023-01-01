#pragma once

#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroInvocationSemi : public Statement {
  mlir::Location location;

public:
};

} // namespace rust_compiler::ast
