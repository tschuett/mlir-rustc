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

  std::shared_ptr<Expression> getName() const { return path; }

  bool hasStructBase() const {
    return expr.has_value() && std::holds_alternative<StructBase>(*expr);
  }
  bool hasStructExprFields() const {
    return expr.has_value() && std::holds_alternative<StructExprFields>(*expr);
  }

  StructBase getStructBase() const { return std::get<StructBase>(*expr); }
  StructExprFields getStructExprFields() const {
    return std::get<StructExprFields>(*expr);
  }
};

} // namespace rust_compiler::ast
