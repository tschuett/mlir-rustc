#pragma once

#include "AST/Visiblity.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::Visibility>
tryParseVisibility(std::span<lexer::Token> tokens);

}
