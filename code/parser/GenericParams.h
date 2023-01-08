#pragma once

#include "AST//GenericArgs.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parse {

std::optional<GenericArgs> tryParseGenericArgs(std::span<lexer::Token> tokens);

}
