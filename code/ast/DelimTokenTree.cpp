#include "AST/DelimTokenTree.h"

#include "Lexer/Token.h"
#include "Location.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::ast {

bool DelimTokenTree::isEmpty() const { return trees.size() == 0; }

std::vector<lexer::Token> DelimTokenTree::toTokenStream() {
  std::vector<Token> tokens;

  // FIXME
  Location loc = Location::getEmptyLocation();

  switch (getKind()) {
  case DelimTokenTreeKind::Paren:
    tokens.push_back(Token(loc, TokenKind::ParenOpen));
    break;
  case DelimTokenTreeKind::Square:
    tokens.push_back(Token(loc, TokenKind::SquareOpen));
    break;
  case DelimTokenTreeKind::Brace:
    tokens.push_back(Token(loc, TokenKind::BraceOpen));
    break;
  }

  for (TokenTree &tree : trees) {
    std::vector<Token> stream = tree.toTokenStream();
    tokens.insert(tokens.end(), stream.begin(), stream.end());
  }

  switch (getKind()) {
  case DelimTokenTreeKind::Paren:
    tokens.push_back(Token(loc, TokenKind::ParenClose));
    break;
  case DelimTokenTreeKind::Square:
    tokens.push_back(Token(loc, TokenKind::SquareClose));
    break;
  case DelimTokenTreeKind::Brace:
    tokens.push_back(Token(loc, TokenKind::BraceClose));
    break;
  }

  return tokens;
}

std::vector<lexer::Token> TokenTree::toTokenStream() {
  std::vector<lexer::Token> tokens;
  if ((bool)tree) {
    std::vector<Token> stream = tree->toTokenStream();
    tokens.insert(tokens.end(), stream.begin(), stream.end());
  }
  if (token)
    tokens.push_back(*token);
  return tokens;
}

} // namespace rust_compiler::ast
