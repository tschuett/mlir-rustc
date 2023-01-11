#pragma once

#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

  std::optional<std::shared_ptr<ast::patterns::PatternWithoutRange>>
    tryParseIdentifierPattern(std::span<lexer::Token>);

}
