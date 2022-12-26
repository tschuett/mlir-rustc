#pragma once

#include "AST/ClippyAttribute.h"
#include "AST/InnerAttribute.h"
#include "AST/OuterAttribute.h"
#include "Token.h"

#include <optional>
#include <span>

namespace rust_compiler {

using namespace rust_compiler::ast;

extern std::optional<OuterAttribute>
tryParseOuterAttribute(std::span<Token> tokens);

extern std::optional<InnerAttribute>
tryParseInnerAttribute(std::span<Token> tokens);

extern std::optional<ClippyAttribute>
tryParseClippyAttribute(std::span<Token> tokens);

} // namespace rust_compiler
