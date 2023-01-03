#pragma once

#include "AST/AST.h"
#include "AST/BlockExpression.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionSignature.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class Function : public Node {
  mlir::Location location;
  std::shared_ptr<BlockExpression> body;
  FunctionSignature signature;
  FunctionQualifiers qualifiers;

public:
  FunctionSignature getSignature() const;
  mlir::Location getLocation() const;
  FunctionQualifiers getFunctionQualifiers() const;

  std::shared_ptr<BlockExpression> getBody();

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
