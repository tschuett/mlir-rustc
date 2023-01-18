#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::TuplePatternItems>>
Parser::tryParseTuplePatternItems(std::span<lexer::Token> tokens) {

  assert(false);

  return std::nullopt;
}

} // namespace rust_compiler::parser
