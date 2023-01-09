#include "AST/SelfParam.h"
#include "FunctionParam.h"
#include "SelfParam.h"

#include <llvm/Support/raw_os_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionParameters>
tryParseFunctionParameters(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;
  FunctionParameters params = {tokens.front().getLocation()};

  llvm::errs() << "tryParseFunctionParameters"
               << "\n";

  //  std::optional<std::shared_ptr<ast::SelfParam>> self =
  //  tryParseSelfParam(view); if (self) {
  //    view = view.subspan((*self)->getTokens());
  //    if (view.front().getKind() == lexer::TokenKind::Comma) {
  //
  //    } else {
  //    }
  //  }

  std::optional<ast::FunctionParam> param = tryParseFunctionParam(view);

  llvm::errs() << "tryParseFunctionParameters: " << param.has_value() << "\n";

  if (param) {
    view = view.subspan((*param).getTokens());
    params.addFunctionParam(*param);

    llvm::errs() << "tryParseFunctionParameters: found first"
                 << "\n";

    while (view.size() > 3) {
      if (view.front().getKind() == lexer::TokenKind::Comma) {
        view = view.subspan(1);
        std::optional<ast::FunctionParam> param = tryParseFunctionParam(view);
        if (param) {
          view = view.subspan((*param).getTokens());
          params.addFunctionParam(*param);
        }
      }
    }
    return params;
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
