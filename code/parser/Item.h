#pragma once

#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <string_view>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Item>>
tryParseItem(std::span<lexer::Token> tokens, std::string_view modulePath);

}
