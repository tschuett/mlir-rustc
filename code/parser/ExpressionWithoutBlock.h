#pragma once

#include "AST/Expression.h"
#include "Lexer/Lexer.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionWithoutBlock(std::span<lexer::Token> view);

}
