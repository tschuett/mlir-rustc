#pragma once

#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/Types/QualifiedPathType.h"

#include <vector>

namespace rust_compiler::ast {

class QualifiedPathInExpression : public PathExpression {
  ast::types::QualifiedPathType type;
  std::vector<PathExprSegment> segments;

public:
  QualifiedPathInExpression(Location loc)
      : PathExpression(loc, PathExpressionKind::QualifiedPathInExpression),
        type(loc) {}

  void setType(const ast::types::QualifiedPathType &ty) { type = ty; }

  void addSegment(const PathExprSegment &seg) { segments.push_back(seg); }

  std::vector<PathExprSegment> getSegment() const { return segments; }
  ast::types::QualifiedPathType getType() const { return type; }
};

} // namespace rust_compiler::ast
