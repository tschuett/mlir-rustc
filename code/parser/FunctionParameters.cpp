#include "FunctionParameters.h"

namespace rust_compiler::parser {

std::optional<ast::FunctionParameters>
tryParseFunctionParameters(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
