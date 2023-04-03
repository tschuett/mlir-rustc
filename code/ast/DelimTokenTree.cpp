#include "AST/DelimTokenTree.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::ast {

bool DelimTokenTree::isEmpty() const { return trees.size() == 0; }

std::vector<lexer::Token> DelimTokenTree::toTokenStream() {
  std::vector<Token> tokens;
  for (TokenTree &tree : trees) {
    std::vector<Token> stream = tree.toTokenStream();
    tokens.insert(tokens.end(), stream.begin(), stream.end());
  }
  return tokens;
}

std::vector<lexer::Token> TokenTree::toTokenStream() {
  std::vector<lexer::Token> tokens;
  if (tree != nullptr) {
    std::vector<Token> stream = tree->toTokenStream();
    tokens.insert(tokens.end(), stream.begin(), stream.end());
  }
  return tokens;
}

} // namespace rust_compiler::ast
