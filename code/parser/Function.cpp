#include "Function.h"

namespace rust_compiler::parser {

  std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                              std::string_view modulePath) {

    // FIXME
    return std::nullopt;
  }

} // namespace rust_compiler::parser
