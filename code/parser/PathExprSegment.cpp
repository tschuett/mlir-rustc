
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<PathExprSegment>
Parser::tryPathExprSegment(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  PathExprSegment expr = {view.front().getLocation()};

  std::optional<std::string> exprSegment = tryParsePathIdentSegment(view);
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

} // namespace rust_compiler::parser
