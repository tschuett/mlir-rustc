#include "Utils.h"

void printTokenState(std::span<rust_compiler::lexer::Token> tokens) {

  if (tokens.size() < 4)
    return;

  printf("next token@%zu ", tokens.size());

  if (tokens[0].isIdentifier()) {
    //    printf("%s ", tokens[0].getIdentifier().c_str());
  } else {
    printf("%s ", Token2String(tokens[0].getKind()).c_str());
  }

  if (tokens[1].isIdentifier()) {
    //printf("%s ", tokens[1].getIdentifier().c_str());
  } else {
    printf("%s ", Token2String(tokens[1].getKind()).c_str());
  }

  if (tokens[2].isIdentifier()) {
    //printf("%s ", tokens[2].getIdentifier().c_str());
  } else {
    printf("%s ", Token2String(tokens[2].getKind()).c_str());
  }

  if (tokens[3].isIdentifier()) {
    //printf("%s ", tokens[3].getIdentifier().c_str());
  } else {
    printf("%s ", Token2String(tokens[3].getKind()).c_str());
  }

  printf("\n");
}
