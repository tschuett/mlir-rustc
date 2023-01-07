#pragma once

#include "AST/Expression.h"
#include "AST/LiteralExpression.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseLiteralExpression(std::span<lexer::Token> tokens);

}
