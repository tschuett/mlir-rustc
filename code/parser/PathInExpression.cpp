#include "AST/PathInExpression.h"

#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParsePathInExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  PathInExpression pathExpr = {view.front().getLocation()};

  if (view.front().getKind() == TokenKind::DoubleColon) {
    view = view.subspan(1);
    pathExpr.addDoubleColon();
  }

  std::optional<PathExprSegment> seg = tryParsePathExprSegment(view);
  if (seg) {
    view = view.subspan((*seg).getTokens());
    pathExpr.addSegment(*seg);
  } else {
    return std::nullopt;
  }

  while (view.size() > 1) {
    if (view.front().getKind() == TokenKind::DoubleColon) {
      pathExpr.addDoubleColon();
      view = view.subspan(1);
      std::optional<PathExprSegment> seg = tryParsePathExprSegment(view);
      if (seg) {
        view = view.subspan((*seg).getTokens());
        pathExpr.addSegment(*seg);
      } else {
        return std::static_pointer_cast<ast::Expression>(
            std::make_shared<PathInExpression>(pathExpr));
      }
    } else {
      return std::static_pointer_cast<ast::Expression>(
          std::make_shared<PathInExpression>(pathExpr));
    }
  }

  return std::static_pointer_cast<ast::Expression>(
      std::make_shared<PathInExpression>(pathExpr));
}

} // namespace rust_compiler::parser
