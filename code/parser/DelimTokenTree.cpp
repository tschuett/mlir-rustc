#include "AST/DelimTokenTree.h"

#include "AST/TokenTree.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::TokenTree> Parser::parseTokenTree() {
  Location loc = getLocation();
  TokenTree tree = {loc};

  if (checkDelimiters()) {
    StringResult<std::shared_ptr<ast::DelimTokenTree>> delimTokenTree =
        parseDelimTokenTree();
    if (!delimTokenTree) {
      llvm::errs() << "failed to parse delim token tree in token tree: "
                   << delimTokenTree.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tree.setTree(delimTokenTree.getValue());
  }

  tree.setToken(getToken());
  assert(eat(getToken().getKind()));
  return StringResult<ast::TokenTree>(tree);
}

StringResult<std::shared_ptr<ast::DelimTokenTree>>
Parser::parseDelimTokenTree() {
  Location loc = getLocation();
  DelimTokenTree tree = {loc};

  if (check(TokenKind::ParenOpen)) {
    assert(eat(TokenKind::ParenOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            "failed to parse delim token tree (eof)");
      } else if (check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::ParenClose));
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            std::make_shared<DelimTokenTree>(tree));
      } else {
        StringResult<ast::TokenTree> tokenTree = parseTokenTree();
        if (!tokenTree) {
          llvm::errs() << "failed to parse token tree in delim token tree: "
                       << tokenTree.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        tree.addTree(tokenTree.getValue());
      }
    }
  } else if (check(TokenKind::SquareOpen)) {
    assert(eat(TokenKind::SquareOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            "failed to parse delim token tree (eof)");
      } else if (check(TokenKind::SquareClose)) {
        assert(eat(TokenKind::SquareClose));
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            std::make_shared<DelimTokenTree>(tree));
      } else {
        StringResult<ast::TokenTree> tokenTree = parseTokenTree();
        if (!tokenTree) {
          llvm::errs() << "failed to parse token tree in delim token tree: "
                       << tokenTree.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        tree.addTree(tokenTree.getValue());
      }
    }
  } else if (check(TokenKind::BraceOpen)) {
    assert(eat(TokenKind::BraceOpen));
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            "failed to parse delim token tree (eof)");
      } else if (check(TokenKind::BraceClose)) {
        assert(eat(TokenKind::BraceClose));
        return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
            std::make_shared<DelimTokenTree>(tree));
      } else {
        StringResult<ast::TokenTree> tokenTree = parseTokenTree();
        if (!tokenTree) {
          llvm::errs() << "failed to parse token tree in delim token tree: "
                       << tokenTree.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        tree.addTree(tokenTree.getValue());
      }
    }
  }

  llvm::errs() << "failed to parse delim token tree" << "\n";
  llvm::errs() << Token2String(getToken().getKind()) << "\n";

  return StringResult<std::shared_ptr<ast::DelimTokenTree>>(
      "failed to parse delim token tree");
}

} // namespace rust_compiler::parser
