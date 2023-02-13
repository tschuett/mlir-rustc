#pragma once

#include "AST/FunctionQualifiers.h"
#include "AST/FunctionSignature.h"
#include "AST/VisItem.h"

namespace rust_compiler::ast {

class BlockExpression;

class Function : public VisItem {
  std::shared_ptr<BlockExpression> body;
  FunctionSignature signature;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc, std::optional<Visibility> vis)
    : VisItem(loc, VisItemKind::Function, vis), signature(loc),
        qualifiers(loc) {}

  const FunctionSignature &getSignature() const;
  FunctionQualifiers getFunctionQualifiers() const;

  bool hasBody() const;

  std::shared_ptr<BlockExpression> getBody();

  void setSignature(FunctionSignature nature);

  void setBody(std::shared_ptr<BlockExpression> block);

  void setVisibility(Visibility vis);
};

} // namespace rust_compiler::ast

// BlockExpression
