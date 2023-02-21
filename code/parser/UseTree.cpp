#include "AST/UseTree.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "llvm/Support/Error.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::use_tree;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::use_tree::UseTree> Parser::parseUseTree() {
  Location loc = getLocation();

  UseTree tree{loc};
  /*
     *
     ::*
     {}
     ::{}

     SimplePath :: *
     SimplePath :: {}
     SimplePath :: { ...}
     SimplePath
     SimplePath as Id or _
   */

  if (check(TokenKind::Star)) {
    tree.setKind(UseTreeKind::Glob);
    assert(check(TokenKind::Star));
    return tree;
    // *
    // done
  } else if (check(TokenKind::PathSep) && check(TokenKind::Star, 1)) {
    tree.setKind(UseTreeKind::Glob);
    tree.setDoubleColon();
    assert(check(TokenKind::PathSep));
    assert(check(TokenKind::Star));
    return tree;
    // :: *
    // done
  } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen, 1) &&
             check(TokenKind::BraceClose, 2)) {
    // :: { }
    assert(check(TokenKind::PathSep));
    assert(check(TokenKind::BraceOpen));
    assert(check(TokenKind::BraceClose));
    tree.setKind(UseTreeKind::Recursive);
    tree.setDoubleColon();
    return tree;
  } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen, 1)) {
    // :: {
    assert(eat(TokenKind::PathSep));
    assert(eat(TokenKind::BraceOpen));
    tree.setKind(UseTreeKind::Recursive);
    tree.setDoubleColon();
    while (true) {
      llvm::Expected<ast::use_tree::UseTree> useTree = parseUseTree();
      if (auto e = useTree.takeError()) {
        llvm::errs() << "failed to use tree in use tree: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
        tree.addTree(*useTree);
      }
      if (check(TokenKind::BraceClose)) {
        // }
        assert(check(TokenKind::BraceClose));
        // done
        return tree;
      } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
        // , }
        assert(check(TokenKind::Comma));
        assert(check(TokenKind::BraceClose));
        // done
        return tree;
      } else if (check(TokenKind::Comma) && !check(TokenKind::BraceClose, 1)) {
        // ,
        assert(check(TokenKind::Comma));
        // continue
      } else if (check(TokenKind::Eof)) {
        // abort
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse use tree: eof");
      }
    }
  } else if (check(TokenKind::BraceOpen) && check(TokenKind::BraceClose, 1)) {
    // { }
    assert(check(TokenKind::BraceOpen));
    assert(check(TokenKind::BraceClose));
    tree.setKind(UseTreeKind::Recursive);
    return tree;
  } else if (check(TokenKind::BraceOpen)) {
    // {
    assert(check(TokenKind::BraceOpen));
    tree.setKind(UseTreeKind::Recursive);
    // WORK: COPY & PASTE
    while (true) {
      llvm::Expected<ast::use_tree::UseTree> useTree = parseUseTree();
      if (auto e = useTree.takeError()) {
        llvm::errs() << "failed to use tree in use tree: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::BraceClose)) {
        assert(eat(TokenKind::BraceClose));
        // }
        // done
        return tree;
      } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
        assert(eat(TokenKind::Comma));
        assert(eat(TokenKind::BraceClose));
        // , }
        // done
        return tree;
      } else if (check(TokenKind::Comma) && !check(TokenKind::BraceClose, 1)) {
        assert(eat(TokenKind::Comma));
        // ,
        continue;
      } else if (check(TokenKind::Eof)) {
        // abort
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse use tree: eof");
      }
    }
  } else {
    // parse simplepath
    llvm::Expected<ast::SimplePath> simple = parseSimplePath();
    if (auto e = simple.takeError()) {
      llvm::errs() << "failed to simple block in use tree: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tree.setPath(*simple);
    // check next token
    if (check(TokenKind::PathSep) && check(TokenKind::Star)) {
      // path :: *
      // done
      tree.setDoubleColon();
      tree.setKind(UseTreeKind::Glob);
      return tree;
    } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen)) {
      // path :: {
      assert(eat(TokenKind::PathSep));
      assert(eat(TokenKind::BraceOpen));
      // WORK: COPY & PASTE
      tree.setKind(UseTreeKind::Recursive);
      while (true) {
        llvm::Expected<ast::use_tree::UseTree> useTree = parseUseTree();
        if (auto e = useTree.takeError()) {
          llvm::errs() << "failed to use tree in use tree: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        if (check(TokenKind::BraceClose)) {
          assert(eat(TokenKind::BraceClose));
          // }
          // done
          return tree;
        } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
          assert(eat(TokenKind::Comma));
          assert(eat(TokenKind::BraceClose));
          // , }
          // done
          return tree;
        } else if (check(TokenKind::Comma) &&
                   !check(TokenKind::BraceClose, 1)) {
          assert(eat(TokenKind::Comma));
          // ,
          continue;
        } else if (check(TokenKind::Eof)) {
          // abort
          return createStringError(inconvertibleErrorCode(),
                                   "failed to parse use tree: eof");
        }
      }
    } else if (checkKeyWord(KeyWordKind::KW_AS)) {
      assert(eatKeyWord(KeyWordKind::KW_AS));
      // path as
      tree.setKind(UseTreeKind::Rebinding);
      if (check(TokenKind::Identifier)) {
        // path as identifier
        tree.setIdentifier(getToken().getIdentifier());
        assert(eat(TokenKind::Identifier));
        // done
        return tree;
      } else if (check(TokenKind::Underscore)) {
        tree.setUnderscore();
        assert(eat(TokenKind::Underscore));
        // path as _
        // done
        return tree;
      }
    } else if (!checkKeyWord(KeyWordKind::KW_AS)) {
      // path
      // done
      return tree;
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse use tree");
}

} // namespace rust_compiler::parser
