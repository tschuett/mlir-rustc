#pragma once

#include "AST/AST.h"
#include "AST/PatternWithoutRange.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast::patterns {

enum class LiteralPatternKind {
  True,
  False,
  CharLiteral,
  ByteLiteral,
  StringLiteral,
  RawStringLiteral,
  ByteStringLiteral,
  RawByteStringLiteral,
  IntegerLiteral,
  FloatLiteral
};

class LiteralPattern : public PatternWithoutRange {
  LiteralPatternKind kind;
  std::string storage;
  bool leadingMinus = false;

public:
  LiteralPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::LiteralPattern) {}

  void setKind(LiteralPatternKind k, std::string_view s) {
    kind = k;
    storage = s;
  }

  void setLeadingMinus() { leadingMinus = true; }
};

} // namespace rust_compiler::ast::patterns
