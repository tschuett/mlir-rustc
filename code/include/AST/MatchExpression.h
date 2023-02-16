#pragma once

#include "AST/Expression.h"
#include "AST/InnerAttribute.h"
#include "AST/MatchArms.h"
#include "AST/Scrutinee.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class Scrutinee;

class MatchExpression : public ExpressionWithBlock {
  Scrutinee scrutinee;
  std::vector<InnerAttribute> innerAttributes;

  std::unique_ptr<MatchArms> matchArms;

public:
  MatchExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::MatchExpression),
        scrutinee(loc) {}

  void setScrutinee(Scrutinee);
};

} // namespace rust_compiler::ast
