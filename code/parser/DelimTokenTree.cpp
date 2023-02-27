#include "AST/DelimTokenTree.h"

#include "AST/TokenTree.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::TokenTree> Parser::parseTokenTree() {
  Location loc = getLocation();
  TokenTree tree = {loc};

  if (checkDelimiters()) {
    llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> delimTokenTree = parseDelimTokenTree();
    if (auto e = delimTokenTree.takeError()) {
      llvm::errs() << "failed to parse delim token tree in token tree : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tree.setTree(*delimTokenTree);
  } else {
    tree.setToken(getToken());
    assert(eat(getToken().getKind()));
    return tree;
  }
}

llvm::Expected<std::shared_ptr<ast::DelimTokenTree>>
Parser::parseDelimTokenTree() {
  Location loc = getLocation();
  DelimTokenTree tree = {loc};

  if (check(TokenKind::ParenOpen)) {
    assert(eat(TokenKind::ParenOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse delim token tree");
      } else if (check(TokenKind::ParenClose)) {
        return std::make_shared<DelimTokenTree>(tree);
      } else {
        llvm::Expected<ast::TokenTree> tokenTree = parseTokenTree();
        if (auto e = tokenTree.takeError()) {
          llvm::errs() << "failed to parse token tree in delim token tree : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        tree.addTree(*tokenTree);
      }
    }
  } else if (check(TokenKind::SquareOpen)) {
    assert(eat(TokenKind::SquareOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse delim token tree");
      } else if (check(TokenKind::SquareClose)) {
        return std::make_shared<DelimTokenTree>(tree);
      } else {
        llvm::Expected<ast::TokenTree> tokenTree = parseTokenTree();
        if (auto e = tokenTree.takeError()) {
          llvm::errs() << "failed to parse token tree in delim token tree : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        tree.addTree(*tokenTree);
      }
    }
  } else if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse delim token tree");
      } else if (check(TokenKind::BraceClose)) {
        return std::make_shared<DelimTokenTree>(tree);
      } else {
        llvm::Expected<ast::TokenTree> tokenTree = parseTokenTree();
        if (auto e = tokenTree.takeError()) {
          llvm::errs() << "failed to parse token tree in delim token tree : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        tree.addTree(*tokenTree);
      }
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse delim token tree");
}

} // namespace rust_compiler::parser
