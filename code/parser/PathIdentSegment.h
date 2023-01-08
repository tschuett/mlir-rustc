#pragma once

#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <string>

namespace rust_compiler::parser {

std::optional<std::string>
tryParsePathIdentSegment(std::span<lexer::Token> tokens);

}
