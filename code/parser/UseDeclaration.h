#pragma once

#include "AST/UseDeclaration.h"

#include <optional>
#include <span>

namespace rust_compiler::ast {

sextern std::optional<UseDeclaration>
tryParseUseDeclaration(std::span<Token> tokens);

}
