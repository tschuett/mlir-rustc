#pragma once

#include "AST/Item.h"
#include "AST/BlockExpression.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionSignature.h"

namespace rust_compiler::ast {

// class FunctionReturnType {};

class Function : public Item {
  std::shared_ptr<ExpressionWithBlock> body;
  FunctionSignature signature;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc) : Item(loc) {}

  FunctionSignature getSignature() const;
  Location getLocation() const;
  FunctionQualifiers getFunctionQualifiers() const;

  std::shared_ptr<ExpressionWithBlock> getBody();

  size_t getTokens() override;

  void setSignature(FunctionSignature nature);

  void setBody(std::shared_ptr<ExpressionWithBlock> block);
};

} // namespace rust_compiler::ast


// BlockExpression
