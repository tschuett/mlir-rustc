#pragma once

#include "AST/BlockExpression.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::BlockExpression>>
tryParseBlockExpression(std::span<lexer::Token> tokens);

} // namespace rust_compiler::parser
