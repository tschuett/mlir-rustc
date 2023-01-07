#pragma once

#include "Lexer/UnicodeCharSets.h"

namespace rust_compiler::lexer {

class CodePoint {
  uint32_t value;

public:
  CodePoint() : value(0) {}
  CodePoint(uint32_t value) : value(value) {}
  bool isspace() const;
  bool isdigit() const;
  bool isxdigit() const;
  bool operator==(char x) { return value == static_cast<uint32_t>(x); }
  bool operator!=(char x) { return value != static_cast<uint32_t>(x); }
  bool operator==(CodePoint x) { return value == x.value; }
  bool operator!=(CodePoint x) { return value != x.value; }
};

} // namespace rust_compiler::lexer
