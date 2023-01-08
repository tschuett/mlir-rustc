#pragma once

#include "Lexer/Token.h"

#include <span>

void printTokenState(std::span<rust_compiler::lexer::Token> tokens);
