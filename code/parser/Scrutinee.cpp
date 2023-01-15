#include "AST/Scrutinee.h"
#include "Parser/Parser.h"
#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Scrutinee>>
Parser::tryParseScrutinee(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> expr =
      tryParseExpression(view);

  if(expr) {
    Scrutinee scrut = {tokens.front().getLocation()};
    scrut.setExpression(*expr);
    return std::make_shared<ast::Scrutinee>(scrut);
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
