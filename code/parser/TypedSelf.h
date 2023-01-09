#pragma once

#include "AST/SelfParam.h"

#include "Lexer/Token.h"

#include <optional>
#include <span>
#include <memory>

namespace rust_compiler::parser {


std::optional<std::shared_ptr<ast::SelfParam>>
  tryParseTypedSelf(std::span<lexer::Token> tokens);


 }
