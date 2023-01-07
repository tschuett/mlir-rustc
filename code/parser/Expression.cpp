#include "Expression.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpression(std::span<lexer::Token>) {

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
