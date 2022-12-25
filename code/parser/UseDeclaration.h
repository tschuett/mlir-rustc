#pragma once

#include "AST/UseDeclaration.h"
#include "Token.h"

#include <optional>
#include <span>

namespace rust_compiler::ast {

extern std::optional<UseDeclaration>
tryParseUseDeclaration(std::span<Token> tokens);

}
