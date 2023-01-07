#include "OperatorExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseOperatorExpression(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

}
