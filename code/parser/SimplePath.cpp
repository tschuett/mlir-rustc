#include "SimplePath.h"

#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static std::optional<ast::SimplePathSegment>
tryParseSimplePathSegment(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().isIdentifier()) {
    return ast::SimplePathSegment(view.front().getLocation(), view.front().getIdentifier());
  }

  return std::nullopt;
}

  std::optional<ast::SimplePath> Parser::tryParseSimplePath(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  ast::SimplePath simplePath{view.front().getLocation()};

  //printf("tryParseSimplePath\n");
  //printTokenState(view);

  if ((view.front().getKind() != TokenKind::DoubleColon) and
      (view.front().getKind() != TokenKind::Identifier))
    return std::nullopt;

  if (view.front().getKind() == TokenKind::DoubleColon) {
    simplePath.setWithDoubleColon();
    view = view.subspan(1);
  }

  //printf("tryParseSimplePath2\n");

  std::optional<ast::SimplePathSegment> simplePathSegment =
      tryParseSimplePathSegment(view);

  if (simplePathSegment) {
    view = view.subspan((*simplePathSegment).getTokens());

    //printf("tryParseSimplePath3: %s\n",
    //       (*simplePathSegment).getSegment().c_str());

    simplePath.addPathSegment(*simplePathSegment);

    if (not view.front().isDoubleColon())
      return simplePath;

    while (view.front().isDoubleColon() && view.size() > 1) {
      view = view.subspan(1); // ::
      std::optional<ast::SimplePathSegment> simplePathSegment =
          tryParseSimplePathSegment(view);
      if (simplePathSegment) {
        //printf("tryParseSimplePath3b: %s\n",
        //       (*simplePathSegment).getSegment().c_str());

        view = view.subspan((*simplePathSegment).getTokens());
        simplePath.addPathSegment(*simplePathSegment);
        if (not view.front().isDoubleColon()) {
          return simplePath;
        }
      } else {
        return simplePath;
      }
    }
  }

  //printf("tryParseSimplePath4\n");
  return std::nullopt;
}

} // namespace rust_compiler::parser
