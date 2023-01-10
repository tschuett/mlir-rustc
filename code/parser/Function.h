#pragma once

#include "AST/Function.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <string_view>

namespace rust_compiler::parser {

std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                              std::string_view modulePath);

std::optional<ast::FunctionQualifiers>
tryParseFunctionQualifiers(std::span<lexer::Token> tokens);

std::optional<ast::FunctionSignature>
tryParseFunctionSignature(std::span<lexer::Token> tokens);

std::optional<std::shared_ptr<ast::types::Type>>
tryParseFunctionReturnType(std::span<lexer::Token> tokens);

} // namespace rust_compiler::parser
