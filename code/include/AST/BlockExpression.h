#pragma once

#include "AST/AST.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class BlockExpression : public Node {
  mlir::Location location;

public:
};

} // namespace rust_compiler::ast
