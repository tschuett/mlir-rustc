#pragma once

#include "AST/AST.h"
#include "AST/FunctionSignature.h"
#include "AST/BlockExpression.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class Function : public Node {
  mlir::Location location;

public:
  FunctionSignature getSignature();
  mlir::Location getLocation();

  std::shared_ptr<BlockExpression> getBody();
};

} // namespace rust_compiler::ast
