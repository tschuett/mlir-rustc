#pragma once

#include "AST/NegationExpression.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseNegationExpression(std::span<lexer::Token> tokens);

}
