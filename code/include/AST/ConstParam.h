#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/Types/TypeExpression.h"
#include "Lexer/Identifier.h"

#include <memory>
#include <optional>
#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class ConstParam : public Node {
  Identifier identifier;
  std::shared_ptr<ast::types::TypeExpression> type;
  std::optional<lexer::Identifier> init;
  std::optional<std::shared_ptr<ast::Expression>> block;
  std::optional<std::shared_ptr<ast::Expression>> literal;

public:
  ConstParam(Location loc) : Node(loc) {}

  void setIdentifier(const lexer::Identifier &i) { identifier = i; }

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }
  void setInit(const Identifier &i) { init = i; }
  void setBlock(std::shared_ptr<ast::Expression> b) { block = b; }
  void setInitLiteral(std::shared_ptr<ast::Expression> lit) { literal = lit; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }

  bool hasLiteral() const { return literal.has_value(); }
  bool hasBlock() const { return block.has_value(); }

  lexer::Identifier getIdentifier() const { return identifier; }
  std::shared_ptr<ast::Expression> getBlock() const { return *block; }
  std::shared_ptr<ast::Expression> getLiteral() const { return *literal; }
};

} // namespace rust_compiler::ast
