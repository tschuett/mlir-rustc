#pragma once

#include "AST/Expression.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

enum class LiteralExpressionKind {
  CharLiteral,
  StringLiteral,
  RawStringLiteral,
  ByteLiteral,
  ByteStringLiteral,
  RawByteStringLiteral,
  IntegerLiteral,
  FloatLiteral,
  True,
  False
};

class LiteralExpression : public ExpressionWithoutBlock {
  LiteralExpressionKind kind;
  std::string value;

public:
  LiteralExpression(Location loc, LiteralExpressionKind kind,
                    std::string_view value)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::LiteralExpression),
        kind(kind), value(value) {}

  LiteralExpressionKind getLiteralKind() const { return kind; }

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;

  std::string getValue() const { return value; }
};

} // namespace rust_compiler::ast
