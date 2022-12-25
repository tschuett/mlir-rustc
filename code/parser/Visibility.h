#pragma once

#include "AST/Visiblity.h"

#include <optional>
#include <span>
#include "Token.h"

namespace rust_compiler::ast {

extern std::optional<Visibility> tryParseVisibility(std::span<Token> tokens);

}
