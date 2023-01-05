#pragma once

#include "AST/Patterns/Patterns.h"

#include "Lexer/Token.h"

#include <span>
#include <optional>
#include <memory>

namespace rust_compiler::lexer {

std::optional<std::shared_ptr<ast::patterns::Pattern>>
tryParsePattern(std::span<lexer::Token> tokens);

}
