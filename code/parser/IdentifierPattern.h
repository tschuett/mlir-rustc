#pragma once

#include "AST/IdentifierPattern.h"
#include "AST/PatternWithoutRange.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::PatternWithoutRange>>
    tryParseIdentifierPattern(std::span<lexer::Token>);

}
