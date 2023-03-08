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
  default: {
    llvm::outs() << "unknown token: " << Token2String(token.getKind()) << "\n";
    exit(EXIT_FAILURE);
  }
  }
}

} // namespace rust_compiler::parser
