#include "AST/SelfParam.h"
#include "FunctionParam.h"
#include "Lexer/Token.h"
#include "llvm/Support/raw_ostream.h"

#include "Parser/Parser.h"

#include <llvm/Support/raw_os_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<ast::FunctionParameters>
Parser::tryParseFunctionParameters(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;
  FunctionParameters params = {tokens.front().getLocation()};

//  llvm::errs() << "tryParseFunctionParameters"
//               << "\n";
//
  // printTokenState(view);

  //  std::optional<std::shared_ptr<ast::SelfParam>> self =
  //  tryParseSelfParam(view); if (self) {
  //    view = view.subspan((*self)->getTokens());
  //    if (view.front().getKind() == lexer::TokenKind::Comma) {
  //
  //    } else {
  //    }
  //  }

  std::optional<ast::FunctionParam> param = tryParseFunctionParam(view);

  //  llvm::errs() << "tryParseFunctionParameters: " << param.has_value() << "\n";

  if (param) {
    view = view.subspan((*param).getTokens());
    params.addFunctionParam(*param);

    //llvm::errs() << "tryParseFunctionParameters: found first"
    //             << "\n";

    if (view.front().getKind() == TokenKind::ParenClose) {
      return params;
    }
    size_t old = view.size();
    while (view.size() > 3) {
      if (view.front().getKind() == lexer::TokenKind::Comma) {
        view = view.subspan(1);
        std::optional<ast::FunctionParam> param = tryParseFunctionParam(view);
        if (param) {
//          llvm::errs() << "tryParseFunctionParameters: found param"
//                       << "\n";
          view = view.subspan((*param).getTokens());
          params.addFunctionParam(*param);
        }
      } else if (view.front().getKind() == lexer::TokenKind::ParenClose) {
        return params;
      }
      if (old == view.size()) {
        llvm::errs() << "no progress: " << old << "\n";
        exit(EXIT_FAILURE);
      }
    }
    return params;
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
