#pragma once

#include "AST/Patterns/IdentifierPattern.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<patterns::IdentifierPattern>
    tryParseIdentifierPattern(std::span<lexer::Token>);

}
