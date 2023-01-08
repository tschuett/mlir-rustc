#pragma once

#include "AST/BlockExpression.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionSignature.h"
#include "AST/Item.h"

namespace rust_compiler::ast {

// class FunctionReturnType {};

class Function : public Item {
  std::shared_ptr<BlockExpression> body;
  FunctionSignature signature;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc) : Item(loc), signature(loc), qualifiers(loc) {}

  FunctionSignature getSignature() const;
  Location getLocation() const;
  FunctionQualifiers getFunctionQualifiers() const;

  std::shared_ptr<BlockExpression> getBody();

  size_t getTokens() override;

  void setSignature(FunctionSignature nature);

  void setBody(std::shared_ptr<BlockExpression> block);
};

} // namespace rust_compiler::ast

// BlockExpression
