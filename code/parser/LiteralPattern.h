#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
tryParseLiteralPattern(std::span<lexer::Token> tokens);

}
