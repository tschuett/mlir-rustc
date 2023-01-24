#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

void Parser::printToken(Token &token) {
  if (token.isIdentifier()) {
    llvm::errs() << "identifier: " << token.getIdentifier() << "\n";
  } else if (token.isKeyWord()) {
    llvm::errs() << "keyword: " << KeyWord2String(token.getKeyWordKind())
                 << "\n";
  } else {
    llvm::errs() << Token2String(token.getKind()) << "\n";
  }
}

} // namespace rust_compiler::parser
