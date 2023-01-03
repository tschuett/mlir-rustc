#pragma once

#include "Lexer/Token.h"

#include "AST/SimplePath.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

  std::optional<ast::SimplePath> tryParseSimplePath(std::span<lexer::Token> tokens);

}
