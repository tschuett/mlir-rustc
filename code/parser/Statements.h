#pragma once

#include "AST/Statements.h"

#include "Lexer/Lexer.h"

#include <optional>
#include <memory>
#include <span>


namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statements>>
tryParseStatements(std::span<lexer::Token> tokens);

}
