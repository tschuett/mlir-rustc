#pragma once

#include "AST/Expression.h"
#include "AST/ReturnExpression.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseReturnExpression(std::span<lexer::Token> tokens);

}
