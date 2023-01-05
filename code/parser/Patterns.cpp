#include "Patterns.h"

namespace rust_compiler::lexer {

  std::optional < std::shared_ptr<ast::patterns::Pattern>>
  tryParsePattern(std::span<lexer::Token> tokens) {
    // FIXME
    return std::nullopt;
  }

}
