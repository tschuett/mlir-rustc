#include "Util.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::parser {

void printTokenState(std::span<lexer::Token> tokens) {

  if (tokens.size() < 4)
    return;

  llvm::errs() << "next token@%" << tokens.size() << "\n";

  if (tokens[0].isIdentifier()) {
    llvm::errs() << tokens[0].getIdentifier().c_str() << " ";
  } else {
    llvm::errs() << Token2String(tokens[0].getKind()).c_str() << "";
  }

  if (tokens[1].isIdentifier()) {
    llvm::errs() << tokens[1].getIdentifier().c_str() << " ";
  } else {
    llvm::errs() << Token2String(tokens[1].getKind()).c_str() << " ";
  }

  if (tokens[2].isIdentifier()) {
    llvm::errs() << tokens[2].getIdentifier().c_str() << " ";
  } else {
    llvm::errs() << Token2String(tokens[2].getKind()).c_str() << " ";
  }

  if (tokens[3].isIdentifier()) {
    llvm::errs() << tokens[3].getIdentifier().c_str() << " ";
  } else {
    llvm::errs() << Token2String(tokens[3].getKind()).c_str() << " ";
  }

  llvm::errs() << "\n";
}

void printStringSpan(std::span<std::string> lintTokens) {
  for (std::string &s : lintTokens) {
    printf("%s ", s.c_str());
  }
  printf("\n");
}

} // namespace rust_compiler::parser
