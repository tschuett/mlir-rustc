#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast::patterns {

enum class RangePatternBoundKind {
  CharLiteral,
  ByteLiteral,
  MinusIntegerLiteral,
  IntegerLiteral,
  MinusFloatLitera,
  FloatLiteral,
  PathExpression
};

class RangePatternBound : public Node {
  RangePatternBoundKind kind;

  std::string storage;
  std::shared_ptr<ast::Expression> path;

public:
  RangePatternBound(Location loc) : Node(loc) {}

  void setKind(RangePatternBoundKind k) { kind = k; }
  void setStorage(std::string_view s) { storage = s; }
  void setPath(std::shared_ptr<ast::Expression> p) { path = p; }
};

} // namespace rust_compiler::ast::patterns
