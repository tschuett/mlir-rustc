#include "SimplePath.h"

#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/Token.h"

#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static std::optional<ast::SimplePathSegment>
tryParseSimplePathSegment(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().isIdentifier()) {
    return ast::SimplePathSegment(view.front().getIdentifier());
  }
  
  return std::nullopt;
}

std::optional<ast::SimplePath> tryParseSimplePath(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  ast::SimplePath simplePath;

  if ((view.front().getKind() != TokenKind::DoubleColon) and
      (view.front().getKind() != TokenKind::Identifier))
    return std::nullopt;

  if (view.front().getKind() == TokenKind::DoubleColon) {
    simplePath.setWithDoubleColon();
    view = view.subspan(1);
  }

  std::optional<ast::SimplePathSegment> simplePathSegment =
      tryParseSimplePathSegment(view);

  if (simplePathSegment) {
    view = view.subspan((*simplePathSegment).getTokens());

    simplePath.addPathSegment(*simplePathSegment);
    while (view.front().isDoubleColon() && view.size() > 1) {
      view = view.subspan(1); // ::
      std::optional<ast::SimplePathSegment> simplePathSegment =
          tryParseSimplePathSegment(view);
      if (simplePathSegment) {
        view = view.subspan((*simplePathSegment).getTokens());
        simplePath.addPathSegment(*simplePathSegment);
      } else {
        return simplePath;
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
