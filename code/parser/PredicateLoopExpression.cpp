#include "AST/PredicateLoopExpression.h"

#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParsePredicateLoopExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isKeyWord() &&
      view.front().getKeyWordKind() == KeyWordKind::KW_WHILE) {
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
