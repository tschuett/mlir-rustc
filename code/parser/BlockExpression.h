#pragma once

#include "AST/BlockExpression.h"

#include "Lexer/Token.h"

#include <span>
#include <optional>

namespace rust_compiler::lexer {

  std::optional<std::shared_ptr<ast::ExpressionWithBlock>>
tryParseBlockExpression(std::span<lexer::Token> tokens);

} // namespace rust_compiler::lexer
