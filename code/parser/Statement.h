#pragma once

#include "AST/Statement.h"
#include "Lexer/Lexer.h"

#include <memory>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statement>>
    tryParseStatement(std::span<lexer::Token>);

}
