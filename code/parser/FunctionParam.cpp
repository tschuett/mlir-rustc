#include "FunctionParam.h"

#include "PatternNoTopAlt.h"


using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionParam>
tryParseFunctionParam(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::PatternNoTopAlt>> notopalt =
      tryParsePatternNoTopAlt(view);

  if (notopalt) {
    view = view.subspan((*notopalt)->getTokens());

    if (view.front().getKind() == TokenKind::Colon) {
      view = view.subspan(1);

      std::optional<std::shared_ptr<ast::types::Type>> type =
          tryParseType(view);

      if (type) {
        FunctionParam param{tokens.front().getLocation()};

        param.setName(*notopalt);
        param.setType(*type);

        return param;
      }
    }
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
