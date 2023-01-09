#pragma once

#include "AST/SelfParam.h"

#include "Lexer/Token.h"

#include <optional>
#include <span>

namespace rust_compiler::parser {

  std::optional<std::shared_ptr<ast::SelfParam>>
tryParseSelfParam(std::span<lexer::Token> tokens);

}
