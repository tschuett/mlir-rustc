#pragma once

namespace rust_compiler::lexer {

class CodePoint {
  uint32_t value;

public:
  Codepoint() : value(0) {}
  Codepoint(uint32_t value) : value(value) {}
  bool isspace() const;
  bool isdigit() const;
  bool isxdigit() const;
  bool operator==(char x) { return v == static_cast<uint32_t>(x); }
  bool operator!=(char x) { return v != static_cast<uint32_t>(x); }
  bool operator==(CodePoint x) { return v == x.value; }
  bool operator!=(CodePoint x) { return v != x.value; }
};

} // namespace rust_compiler::lexer
