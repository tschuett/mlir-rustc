#include "WhereClause.h"

namespace rust_compiler::parser {

std::optional<ast::WhereClause>
tryParseWhereClause(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
