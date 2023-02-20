#include "AST/UseDeclaration.h"

#include "AST/UseTree.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseUseDeclaration(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  UseDeclaration use = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_USE)) {
    assert(eatKeyWord(KeyWordKind::KW_USE));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse use keyword in use declarion");
  }

  llvm::Expected<ast::use_tree::UseTree> tree = parseUseTree();
  if (auto e = tree.takeError()) {
    llvm::errs() << "failed to use tree in use declaration: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  use.setTree(*tree);

  return std::make_shared<UseDeclaration>(use);
}

llvm::Expected<ast::use_tree::UseTree> Parser::parseUseTree() {

  if (check(TokenKind::Star)) {
    // *
    // done
  } else if (check(TokenKind::PathSep) && check(TokenKind::Star, 1)) {
    // :: *
    // done
  } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen, 1) &&
             check(TokenKind::BraceClose, 2)) {
    // :: { }
  } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen, 1)) {
    // :: {
    assert(eat(TokenKind::PathSep));
    assert(eat(TokenKind::BraceOpen));
    while (true) {
      llvm::Expected<ast::use_tree::UseTree> useTree = parseUseTree();
      if (auto e = useTree.takeError()) {
        llvm::errs() << "failed to use tree in use tree: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::BraceClose)) {
        // }
        // done
      } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
        // , }
        // done
      } else if (check(TokenKind::Comma) && !check(TokenKind::BraceClose, 1)) {
        // ,
        // continue
      } else if (check(TokenKind::Eof)) {
        // abort
      }
    }
  } else if (check(TokenKind::BraceOpen)) {
    // {
  } else {
    // parse simplepath
    llvm::Expected<ast::SimplePath> simple = parseSimplePath();
    if (auto e = simple.takeError()) {
      llvm::errs() << "failed to simple block in use tree: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    // check next token
    if (check(TokenKind::PathSep) && check(TokenKind::Star)) {
      // path :: *
      // done
    } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen)) {
      // path :: {
      assert(eat(TokenKind::PathSep));
      assert(eat(TokenKind::BraceOpen));
      // WORK: COPY & PASTE
      while (true) {
        llvm::Expected<ast::use_tree::UseTree> useTree = parseUseTree();
        if (auto e = useTree.takeError()) {
          llvm::errs() << "failed to use tree in use tree: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        if (check(TokenKind::BraceClose)) {
          // }
          // done
        } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
          // , }
          // done
        } else if (check(TokenKind::Comma) &&
                   !check(TokenKind::BraceClose, 1)) {
          // ,
          continue;
        } else if (check(TokenKind::Eof)) {
          // abort
        }
      }
    } else if (checkKeyWord(KeyWordKind::KW_AS)) {
      // path as
      if (check(TokenKind::Identifier)) {
        // path as identifier
        // done
      } else if (check(TokenKind::Underscore)) {
        // path as _
        // done
      }
    } else if (!checkKeyWord(KeyWordKind::KW_AS)) {
      // path
      // done
    }
  }
}

} // namespace rust_compiler::parser
