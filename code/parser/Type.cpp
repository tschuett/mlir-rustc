#include "Type.h"

#include "PrimitiveType.h"

#include "Parser/Parser.h"

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
Parser::tryParseType(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::types::Type>> primitiveType =
      tryParsePrimitiveType(view);

  if (primitiveType) {
    return primitiveType;
  }

    // FIXME


  return std::nullopt;
}

} // namespace rust_compiler::parser
