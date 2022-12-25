#pragma once

#include "AST/InnerAttribute.h"
#include "AST/OuterAttribute.h"

#include <optional>
#include <span>

#include "Token.h"

namespace rust_compiler {

using namespace rust_compiler::ast;

extern std::optional<OuterAttribute>
tryParseOuterAttribute(std::span<Token> tokens);

extern std::optional<InnerAttribute>
tryParseInnerAttribute(std::span<Token> tokens);

} // namespace rust_compiler
