#include "Parser/Precedence.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

Precedence getLeftBindingPower(lexer::Token token) {
  switch (token.getKind()) {
  case TokenKind::PathSep:
    return Precedence::Path;
    
  }
}

} // namespace rust_compiler::parser
