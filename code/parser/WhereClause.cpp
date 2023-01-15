#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<ast::WhereClause>
Parser::tryParseWhereClause(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
