#pragma once

#include "AST/PathInExpression.h"
#include "AST/StructBase.h"
#include "AST/StructExprFields.h"
#include "AST/StructExpression.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class StructExprStruct : public StructExpression {
  PathInExpression path;
  std::optional<std::variant<StructExprFields, StructBase>> expr;

public:
  StructExprStruct(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprStruct) {}
};

} // namespace rust_compiler::ast
