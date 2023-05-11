#pragma once

#include "AST/Expression.h"
#include "AST/OuterAttribute.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class StructExprField : public Node {
  std::vector<OuterAttribute> outer;

  std::optional<std::string> identifier;
  std::optional<std::string> tupleIndex;

  std::shared_ptr<Expression> expr;

public:
  StructExprField(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outer = {o.begin(), o.end()};
  }

  void setIdentifier(std::string_view s) { identifier = s; }
  void setTupleIndex(std::string_view t) { tupleIndex = t; }
  void setExpression(std::shared_ptr<ast::Expression> e) { expr = e; }

  bool hasExpression() const { return (bool)expr; }

  std::shared_ptr<Expression> getExpression() const { return expr; }
};

} // namespace rust_compiler::ast
