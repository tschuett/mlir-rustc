#pragma once

#include "Lexer/Token.h"

#include <span>

void printTokenState(std::span<lexer::Token> tokens);
