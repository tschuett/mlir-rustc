#include "Parser/Parser.h"

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<GenericArgs>
Parser::tryParseGenericArgs(std::span<lexer::Token> tokens) {

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
