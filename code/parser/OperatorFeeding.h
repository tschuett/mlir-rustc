#pragma once

#include "AST/Expression.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseOperatorFeedingExpression(std::span<lexer::Token> tokens);

}