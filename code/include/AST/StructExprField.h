#pragma once

#include "AST/Expression.h"
#include "AST/OuterAttribute.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class StructExprField : public Node {
  std::vector<OuterAttribute> outer;

  std::variant<std::string, uint32_t> name;

  std::shared_ptr<Expression> expr;

public:
  StructExprField(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
