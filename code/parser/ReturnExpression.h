#pragma once

#include "AST/ReturnExpression.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::ReturnExpression>
tryParseReturnExpression(std::span<lexer::Token> tokens);

}
