#include "PathInExpression.h"

#include "AST/GenericArgs.h"
#include "AST/PathExprSegment.h"
#include "AST/PathInExpression.h"
#include "Lexer/KeyWords.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<GenericArgs> tryParseGenericArgs(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

std::optional<std::string> tryPathIdentSegment(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().isIdentifier())
    return view.front().getIdentifier();

  if (view.front().isKeyWord()) {
    std::string key = view.front().getIdentifier();
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SUPER)
      return std::string("super");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SELFVALUE)
      return std::string("self");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_SELFTYPE)
      return std::string("Self");
    if (view.front().getKeyWordKind() == lexer::KeyWordKind::KW_CRATE)
      return std::string("crate");
  }

  // FIXME
  return std::nullopt;
}

std::optional<PathExprSegment>
tryPathExprSegment(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  PathExprSegment expr = {view.front().getLocation()};

  std::optional<std::string> exprSegment = tryPathIdentSegment(view);
  if (exprSegment) {
    view = view.subspan(1);

    expr.addIdentSegment(*exprSegment);
    while (view.size() > 1) {
      if (view.front().getKind() == TokenKind::DoubleColon) {
        view = view.subspan(1);
        std::optional<GenericArgs> genericArgs = tryParseGenericArgs(view);
        if (genericArgs) {
          expr.addGenerics(*genericArgs);
        } else {
          return expr;
        }
      }
      return expr;
    }
    return expr;
  }
  // FIXME
  return std::nullopt;
}

std::optional<std::shared_ptr<ast::Expression>>
tryParsePathInExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  PathInExpression pathExpr = {view.front().getLocation()};

  if (view.front().getKind() == TokenKind::DoubleColon) {
    view = view.subspan(1);
  }

  std::optional<PathExprSegment> seg = tryPathExprSegment(view);
  if (seg) {
    view = view.subspan((*seg).getTokens());
    pathExpr.addSegment(*seg);
  } else {
    return std::nullopt;
  }

  while (view.size() > 1) {
    if (view.front().getKind() == TokenKind::DoubleColon) {
      view = view.subspan(1);
      std::optional<PathExprSegment> seg = tryPathExprSegment(view);
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
