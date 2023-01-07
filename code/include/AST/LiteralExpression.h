#pragma once

#include "AST/Expression.h"

#include <string_view>
#include <string>

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
 LiteralExpression(Location loc, LiteralExpressionKind kind, std::string_view value)
   : ExpressionWithoutBlock(loc), kind(kind), value(value) {}

  size_t getTokens() override;

  LiteralExpressionKind getLiteralKind() const { return kind; }
};

} // namespace rust_compiler::ast
