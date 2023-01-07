#pragma once

#include "AST/Expression.h"
#include "AST/PathExpression.h"

#include <optional>
#include <span>
#include <memory>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParsePathInExpression(std::span<lexer::Token> tokens);

}
