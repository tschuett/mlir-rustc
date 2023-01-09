#pragma once

#include "AST/ShorthandSelf.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::SelfParam>>
tryParseShorthandSelf(std::span<lexer::Token> tokens);

}
