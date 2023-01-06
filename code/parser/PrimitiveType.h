#pragma once

#include "AST/Types/Types.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
tryParsePrimitiveType(std::span<lexer::Token> tokens);

}
