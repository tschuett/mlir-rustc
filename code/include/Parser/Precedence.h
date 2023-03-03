#pragma once

#include "Lexer/Token.h"

namespace rust_compiler::parser {

enum class Precedence {
  Highest = 100,
  Paths = 95,
  MethodCall = 90,
  FieldExpression = 85,
  FunctionCall = 80,
  ArrayIndexing = 80,
  QuestionMark = 75,

  Equal = 30,
  NotEqual = 30,
  GreaterThan = 30,
  LessThan = 30,
  GreaterThanOrEqualTo = 30,
  LessThanOrEqualTo = 30,

  Lowest = 0
};

int getLeftBindingPower(lexer::Token);

} // namespace rust_compiler::parser
