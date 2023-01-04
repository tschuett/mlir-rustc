#pragma once

#include <AST/Types/Types.h>
#include <Lexer/Token.h>

namespace rust_compiler::lexer {

std::optional<std::shared_ptr<ast::Type>>
tryParseType(std::span<lexer::Token> tokens);

}
