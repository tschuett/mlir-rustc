#pragma once

#include "AST/PathInExpression.h"
#include "AST/StructBase.h"
#include "AST/StructExprFields.h"
#include "AST/StructExpression.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class StructExprStruct : public StructExpression {
  std::shared_ptr<Expression> path;
  std::optional<std::variant<StructExprFields, StructBase>> expr;

public:
  StructExprStruct(Location loc)
    : StructExpression(loc, StructExpressionKind::StructExprStruct) {}

  void setPath(std::shared_ptr<Expression> p) { path = p; }
  void setBase(const StructBase &b) { expr = b; }
  void setFields(const StructExprFields &f) { expr = f; }
};

} // namespace rust_compiler::ast
