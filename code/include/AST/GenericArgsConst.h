#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/SimplePathSegment.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

enum class GenericArgsConstKind {
  BlockExpression,
  LiteralExpression,
  SimplePathSegment
};

class GenericArgsConst : public Node {
  // GenericArgsConstKind kind;

  std::shared_ptr<Expression> blockExpression;
  std::shared_ptr<Expression> literalExpression;
  std::optional<SimplePathSegment> segment;

  bool hasLeadingMinus = false;

public:
  GenericArgsConst(Location loc) : Node(loc) {}

  void setLeadingMinus() { hasLeadingMinus = true; }

  void setBlock(std::shared_ptr<Expression> b) { blockExpression = b; }
  void setLiteral(std::shared_ptr<Expression> l) { literalExpression = l; }
  void setSegment(const SimplePathSegment &p) { segment = p; }
};

} // namespace rust_compiler::ast
