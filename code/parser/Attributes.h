#pragma once

#include "AST/ClippyAttribute.h"
#include "AST/InnerAttribute.h"
#include "AST/OuterAttribute.h"
#include "Lexer/Token.h"

#include <mlir/IR/Location.h>
#include <optional>
#include <span>

namespace rust_compiler::parser {

using namespace rust_compiler::ast;

std::optional<OuterAttribute>
tryParseOuterAttribute(std::span<lexer::Token> tokens);

std::optional<InnerAttribute>
tryParseInnerAttribute(std::span<lexer::Token> tokens);

std::optional<ClippyAttribute>
tryParseClippyAttribute(std::span<lexer::Token> tokens);

} // namespace rust_compiler::parser
