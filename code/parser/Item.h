#pragma once

#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <string_view>

namespace rust_compiler::parser {

std::optional<ast::Item> tryParseItem(std::span<lexer::Token> tokens,
                                      std::string_view modulePath);

}
