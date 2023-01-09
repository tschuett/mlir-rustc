#pragma once

#include "AST/FunctionParam.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<ast::FunctionParam>
tryParseFunctionParam(std::span<lexer::Token> tokens);

}
