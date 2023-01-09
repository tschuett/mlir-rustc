#pragma once

#include "AST/AST.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

class NegationExpression : public OperatorExpression {
  bool notToken = false;
  bool minusToken = false;
  std::shared_ptr<Expression> right;

public:
  NegationExpression(Location loc) : OperatorExpression(loc){};

  void setRight(std::shared_ptr<Expression>);
  void setMinus();
  void setNot();

  size_t getTokens() override;
};

} // namespace rust_compiler::ast