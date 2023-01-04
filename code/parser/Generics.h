#pragma once

#include "Lexer/Token.h"

#include "AST/GenericParams.h"

#include <memory>
#include <optional>
#include <span>

namespace rust_compiler::lexer {

  std::optional<std::shared_ptr<ast::GenericParams>>
tryParseGenericParams(std::span<lexer::Token> tokens);

}
