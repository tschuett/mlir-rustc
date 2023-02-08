
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Types/TypeExpression.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<ast::FunctionParam>
Parser::tryParseFunctionParam(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  //  llvm::errs() << "tryParseFunctionParam"
  //               << "\n";

  // std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> notopalt =
  //     tryParsePatternNoTopAlt(view);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>> notopalt =
      tryParseIdentifierPattern(view);

  // llvm::errs() << "tryParseFunctionParam " << notopalt.has_value() << "\n";

  if (notopalt) {
    view = view.subspan((*notopalt)->getTokens());

    // llvm::errs() << "tryParseFunctionParam: tokens: "
    //              << (*notopalt)->getTokens() << "\n";

    if (view.front().getKind() == TokenKind::Colon) {
      view = view.subspan(1);
      // llvm::errs() << "tryParseFunctionParam: found colon"
      //              << "\n";

      // printTokenState(view);

      std::optional<std::shared_ptr<ast::types::TypeExpression>> type =
          tryParseTypeExpression(view);

      if (type) {
        FunctionParam param{tokens.front().getLocation()};

        // llvm::errs() << "tryParseFunctionParam: found type"
        //            << "\n";

        param.setName(
            std::static_pointer_cast<ast::patterns::IdentifierPattern>(
                *notopalt));
        param.setType(*type);

        return param;
      }
    }
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
