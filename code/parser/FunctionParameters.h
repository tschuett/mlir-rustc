#pragma once

#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::FunctionParameters>
tryParseFunctionParameters(std::span<lexer::Token> tokens);

}
