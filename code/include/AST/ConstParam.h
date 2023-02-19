#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Expression.h"

#include <memory>
#include <optional>
#include <string_view>

namespace rust_compiler::ast {

class ConstParam : public Node {
  std::string identifier;
  std::shared_ptr<ast::types::TypeExpression> type;
  std::optional<std::string> init;
  std::optional<std::shared_ptr<ast::Expression>> block;
  std::optional<std::shared_ptr<ast::Expression>> literal;

public:
  ConstParam(Location loc) : Node(loc) {}

  void setIdentifier(std::string_view i) { identifier = i; }

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }
  void setInit(std::string_view i) { init = i; }
  void setBlock(std::shared_ptr<ast::Expression> b) { block = b; }
  void setInitLiteral(std::shared_ptr<ast::Expression> lit) { literal = lit; }
};

} // namespace rust_compiler::ast
