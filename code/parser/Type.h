#pragma once

#include <AST/Types/Types.h>
#include <Lexer/Token.h>

#include <span>
#include <optional>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Type>>
tryParseType(std::span<lexer::Token> tokens);

}
