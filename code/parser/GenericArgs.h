#pragma once

#include "AST//GenericArgs.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::GenericArgs>
tryParseGenericArgs(std::span<lexer::Token> tokens);

}
