#pragma once

#include "AST/BlockExpression.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionSignature.h"
#include "AST/VisItem.h"

namespace rust_compiler::ast {

// class FunctionReturnType {};

class Function : public VisItem {
  std::shared_ptr<BlockExpression> body;
  FunctionSignature signature;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc) : VisItem(loc, VisItemKind::Function), signature(loc), qualifiers(loc) {}

  FunctionSignature getSignature() const;
  Location getLocation() const;
  FunctionQualifiers getFunctionQualifiers() const;

  std::shared_ptr<BlockExpression> getBody();

  size_t getTokens() override;

  void setSignature(FunctionSignature nature);

  void setBody(std::shared_ptr<BlockExpression> block);

  void setVisibility(Visibility vis);
};

} // namespace rust_compiler::ast

// BlockExpression
