#pragma once

#include "AST/PathExprSegment.h"
#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

std::optional<PathExprSegment>
tryPathExprSegment(std::span<lexer::Token> tokens);


}
