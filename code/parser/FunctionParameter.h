#pragma once

#include "AST/FunctionParameter.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::FunctionParameter>
tryParseFunctionParameter(std::span<lexer::Token> tokens);

}
