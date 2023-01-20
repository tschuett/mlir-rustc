#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
Parser::tryParseQualifiedPathInType(std::span<lexer::Token> tokens) {

  assert(false);
}

} // namespace rust_compiler::parser
