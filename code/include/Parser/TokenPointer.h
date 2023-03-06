#pragma once

#include "Lexer/TokenStream.h"

namespace rust_compiler::parser {

class ConstTokenPointer {
  const lexer::TokenStream *stream;
  size_t offset;

public:
  ConstTokenPointer(const lexer::TokenStream *stream, size_t offset);
};

} // namespace rust_compiler::parser
