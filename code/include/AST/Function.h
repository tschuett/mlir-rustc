#pragma once

#include "AST/AST.h"
#include "AST/FunctionSignature.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class Function : public Node {
  mlir::Location location;

public:
  FunctionSignature getSignature();
  mlir::Location getLocation();
};

} // namespace rust_compiler::ast
