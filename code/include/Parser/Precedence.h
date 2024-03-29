#pragma once

#include "Lexer/Token.h"

namespace rust_compiler::parser {

enum class Precedence {
  Highest = 100,
  Path = 95,
  MethodCall = 90,
  FieldExpression = 85,
  FunctionCall = 80,
  ArrayIndexing = 80,
  QuestionMark = 75,

  UnaryAsterisk = 70,
  UnaryAnd = 70,
  UnaryMinus = 70,
  UnaryAndMut = 70,
  UnaryNot = 70,
  UnaryStar = 70,

  As = 65,

  Mul = 60,

  Plus = 55,
  Minus = 55,

  LShift = 50,
  RShift = 50,
  
  Equal = 30,
  NotEqual = 30,
  GreaterThan = 30,
  LessThan = 30,
  GreaterThanOrEqualTo = 30,
  LessThanOrEqualTo = 30,

  DotDot = 15,
  DotDotEq = 15,

  Assign = 10,
  PlusAssign = 10,
  MinusAssign = 10,
  MulAssign = 10,
  DivAssign = 10,
  RemAssign = 10,
  XorAssign = 10,
  AndAssign = 10,
  OrAssign = 10,
  ShlAssign = 10,
  ShrAssign = 10,

  Lowest = 0
};

} // namespace rust_compiler::parser
