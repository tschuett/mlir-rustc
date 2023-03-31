#pragma once

#include <limits>
#include <string>

namespace rust_compiler {

class Span {
  std::string fileName;
  unsigned lowLineNumber;
  unsigned highLineNumber;
  unsigned lowColumnNumber;
  unsigned highColumnNumber;

public:
  Span(std::string_view fileName, unsigned lowLineNumber,
       unsigned highLineNumber, unsigned lowColumnNumber,
       unsigned highColumnNumber)
      : fileName(fileName), lowLineNumber(lowLineNumber),
        highLineNumber(highLineNumber), lowColumnNumber(lowColumnNumber),
        highColumnNumber(highColumnNumber) {}

  static Span getBuiltinSpan() {
    return Span("builtins.cpp", std::numeric_limits<unsigned>::min(),
                std::numeric_limits<unsigned>::max(),
                std::numeric_limits<unsigned>::min(),
                std::numeric_limits<unsigned>::max());
  }

  static Span getEmptySpan() {
    return Span("empty.cpp", std::numeric_limits<unsigned>::min(),
                std::numeric_limits<unsigned>::max(),
                std::numeric_limits<unsigned>::min(),
                std::numeric_limits<unsigned>::max());
  }

  std::string toString() const;
};

} // namespace rust_compiler
