#pragma once

#include "AST/PatternNoTopAlt.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::PatternNoTopAlt>>
tryParsePatternNoTopAlt(std::span<lexer::Token> tokens);

}
