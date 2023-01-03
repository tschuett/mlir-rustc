#pragma once

#include "AST/Function.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <string_view>

namespace rust_compiler::parser {

std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                              std::string_view modulePath);

} // namespace rust_compiler::parser
