#pragma once

#include "Lexer/Token.h"

#include <span>

namespace rust_compiler::parser {

void printTokenState(std::span<lexer::Token> tokens);

void printStringSpan(std::span<std::string> lintTokens);

} // namespace rust_compiler::parser
