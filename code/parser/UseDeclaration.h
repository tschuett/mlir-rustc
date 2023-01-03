#pragma once

#include "AST/UseDeclaration.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<UseDeclaration>
tryParseUseDeclaration(std::span<lexer::Token> tokens);

}
