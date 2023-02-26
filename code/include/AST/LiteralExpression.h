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

class LiteralExpression final : public ExpressionWithoutBlock {
  LiteralExpressionKind kind;
  std::string value;

public:
  LiteralExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::LiteralExpression) {}

  LiteralExpressionKind getLiteralKind() const { return kind; }

  void setKind(LiteralExpressionKind k) { kind = k; }

  std::string getValue() const { return value; }

  void setStorage(std::string_view s) { value = s; }
};

} // namespace rust_compiler::ast
