#pragma once

#include "AST/WhereClause.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::WhereClause>
tryParseWhereClause(std::span<lexer::Token> tokens);

}
