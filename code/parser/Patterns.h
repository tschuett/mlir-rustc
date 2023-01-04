#pragma once

#include "AST/Patterns/Patterns.h"

namespace rust_compiler::lexer {

  std::optional < std::shared_ptr<ast::patterns::Pattern>
                tryParsePattern(std::span<lexer::token> tokens);

}
