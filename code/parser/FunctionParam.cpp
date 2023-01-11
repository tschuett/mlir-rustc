#include "FunctionParam.h"

#include "PatternNoTopAlt.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionParam>
tryParseFunctionParam(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

//  llvm::errs() << "tryParseFunctionParam"
//               << "\n";

  std::optional<std::shared_ptr<ast::PatternNoTopAlt>> notopalt =
      tryParsePatternNoTopAlt(view);

  //llvm::errs() << "tryParseFunctionParam " << notopalt.has_value() << "\n";

  if (notopalt) {
    view = view.subspan((*notopalt)->getTokens());

    //llvm::errs() << "tryParseFunctionParam: tokens: "
    //             << (*notopalt)->getTokens() << "\n";

    if (view.front().getKind() == TokenKind::Colon) {
      view = view.subspan(1);
      //llvm::errs() << "tryParseFunctionParam: found colon"
      //             << "\n";

      printTokenState(view);

      std::optional<std::shared_ptr<ast::types::Type>> type =
          tryParseType(view);

      if (type) {
        FunctionParam param{tokens.front().getLocation()};

        //llvm::errs() << "tryParseFunctionParam: found type"
        //           << "\n";

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
