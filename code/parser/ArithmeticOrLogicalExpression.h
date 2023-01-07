#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseArithmeticOrLogicalExpresion(std::span<lexer::Token> tokens);

}
