#pragma once

#include "AST/PatternNoTopAlt.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::PatternNoTopAlt>>
tryParseLiteralPattern(std::span<lexer::Token> tokens);

}
