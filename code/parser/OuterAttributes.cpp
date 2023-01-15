#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::OuterAttributes>>
Parser::tryParseOuterAttributes(std::span<lexer::Token> tokens,
                                std::string_view modulePath) {

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
