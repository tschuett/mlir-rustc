#pragma once

#include "AST/Expression.h"
#include "Lexer/Token.h"

#include "AST/OperatorExpression.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseOperatorExpression(std::span<lexer::Token> tokens);

}
