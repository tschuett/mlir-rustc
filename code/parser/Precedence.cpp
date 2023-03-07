#include "Parser/Precedence.h"

#include "Lexer/Token.h"

#include <cstdlib>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

Precedence Parser::getLeftBindingPower(lexer::Token token) {
  switch (token.getKind()) {
  case TokenKind::PathSep:
    return Precedence::Path;
  }
default: {
  llvm::outs() << "unknown token: " << Token2String(token.getKind()) << "\n";
  exit(EXIT_FAILURE);
}
}

} // namespace rust_compiler::parser
