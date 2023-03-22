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

  MatchArms matchArms;

public:
  MatchExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::MatchExpression),
        scrutinee(loc), matchArms(loc) {}

  void setScrutinee(const Scrutinee &scru) { scrutinee = scru; }
  void setInnerAttributes(std::span<InnerAttribute> inner) {
    innerAttributes = {inner.begin(), inner.end()};
  }

  void setMatchArms(const MatchArms &ma) { matchArms = ma; }

  Scrutinee getScrutinee() const { return scrutinee; }

  MatchArms getMatchArms() const { return matchArms; }
};

} // namespace rust_compiler::ast
