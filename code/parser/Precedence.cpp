#include "Parser/Precedence.h"

#include "Lexer/Token.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

Precedence getLeftBindingPower(const Token &token) {
  switch (token.getKind()) {
  case TokenKind::PathSep:
    return Precedence::Path;
  case TokenKind::Plus:
    return Precedence::Plus;
  case TokenKind::Minus:
    return Precedence::Minus;
  case TokenKind::Lt:
    return Precedence::LessThan;
  case TokenKind::Eq:
    return Precedence::Equal;
  default: {
    llvm::outs() << "getLeftBindingPower: unknown token: "
                 << Token2String(token.getKind()) << "\n";
    return Precedence::Lowest;
    exit(EXIT_FAILURE);
  }
  }
}

} // namespace rust_compiler::parser
