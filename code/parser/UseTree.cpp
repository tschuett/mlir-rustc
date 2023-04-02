#include "AST/UseTree.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::use_tree;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::use_tree::UseTree> Parser::parseUseTree() {
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
    return StringResult<ast::use_tree::UseTree>(tree);
    // *
    // done
  } else if (check(TokenKind::PathSep) && check(TokenKind::Star, 1)) {
    tree.setKind(UseTreeKind::Glob);
    tree.setDoubleColon();
    assert(check(TokenKind::PathSep));
    assert(check(TokenKind::Star));
    return StringResult<ast::use_tree::UseTree>(tree);
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
    return StringResult<ast::use_tree::UseTree>(tree);
  } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen, 1)) {
    // :: {
    assert(eat(TokenKind::PathSep));
    assert(eat(TokenKind::BraceOpen));
    tree.setKind(UseTreeKind::Recursive);
    tree.setDoubleColon();
    while (true) {
      StringResult<ast::use_tree::UseTree> useTree = parseUseTree();
      if (!useTree) {
        llvm::errs() << "failed to parse use tree in use tree: "
                     << useTree.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tree.addTree(useTree.getValue());

      if (check(TokenKind::BraceClose)) {
        // }
        assert(check(TokenKind::BraceClose));
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
        // , }
        assert(check(TokenKind::Comma));
        assert(check(TokenKind::BraceClose));
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      } else if (check(TokenKind::Comma) && !check(TokenKind::BraceClose, 1)) {
        // ,
        assert(check(TokenKind::Comma));
        // continue
      } else if (check(TokenKind::Eof)) {
        // abort
        return StringResult<ast::use_tree::UseTree>(
            "failed to parse use tree: eof");
      }
    }
  } else if (check(TokenKind::BraceOpen) && check(TokenKind::BraceClose, 1)) {
    // { }
    assert(check(TokenKind::BraceOpen));
    assert(check(TokenKind::BraceClose));
    tree.setKind(UseTreeKind::Recursive);
    return StringResult<ast::use_tree::UseTree>(tree);
  } else if (check(TokenKind::BraceOpen)) {
    // {
    assert(check(TokenKind::BraceOpen));
    tree.setKind(UseTreeKind::Recursive);
    // WORK: COPY & PASTE
    while (true) {
      StringResult<ast::use_tree::UseTree> useTree = parseUseTree();
      if (!useTree) {
        llvm::errs() << "failed to parse use tree in use tree: "
                     << useTree.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::BraceClose)) {
        assert(eat(TokenKind::BraceClose));
        // }
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
        assert(eat(TokenKind::Comma));
        assert(eat(TokenKind::BraceClose));
        // , }
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      } else if (check(TokenKind::Comma) && !check(TokenKind::BraceClose, 1)) {
        assert(eat(TokenKind::Comma));
        // ,
        continue;
      } else if (check(TokenKind::Eof)) {
        // abort
        return StringResult<ast::use_tree::UseTree>(
            "failed to parse use tree: eof");
      }
    }
  } else {
    // parse simplepath
    StringResult<ast::SimplePath> simple = parseSimplePath();
    if (!simple) {
      llvm::errs() << "failed to parse simple block in use tree: "
                   << simple.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tree.setPath(simple.getValue());
    // check next token
    if (check(TokenKind::PathSep) && check(TokenKind::Star)) {
      // path :: *
      // done
      tree.setDoubleColon();
      tree.setKind(UseTreeKind::Glob);
      return StringResult<ast::use_tree::UseTree>(tree);
    } else if (check(TokenKind::PathSep) && check(TokenKind::BraceOpen)) {
      // path :: {
      assert(eat(TokenKind::PathSep));
      assert(eat(TokenKind::BraceOpen));
      // WORK: COPY & PASTE
      tree.setKind(UseTreeKind::Recursive);
      while (true) {
        StringResult<ast::use_tree::UseTree> useTree = parseUseTree();
        if (!useTree) {
          llvm::errs() << "failed to parse use tree in use tree: "
                       << simple.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        if (check(TokenKind::BraceClose)) {
          assert(eat(TokenKind::BraceClose));
          // }
          // done
          return StringResult<ast::use_tree::UseTree>(tree);
        } else if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
          assert(eat(TokenKind::Comma));
          assert(eat(TokenKind::BraceClose));
          // , }
          // done
          return StringResult<ast::use_tree::UseTree>(tree);
        } else if (check(TokenKind::Comma) &&
                   !check(TokenKind::BraceClose, 1)) {
          assert(eat(TokenKind::Comma));
          // ,
          continue;
        } else if (check(TokenKind::Eof)) {
          // abort
          return StringResult<ast::use_tree::UseTree>(
              "failed to parse use tree: eof");
        }
      }
    } else if (checkKeyWord(KeyWordKind::KW_AS)) {
      assert(eatKeyWord(KeyWordKind::KW_AS));
      // path as
      tree.setKind(UseTreeKind::Rebinding);
      if (check(TokenKind::Identifier)) {
        // path as identifier
        tree.setIdentifier(getToken().getIdentifier().toString());
        assert(eat(TokenKind::Identifier));
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      } else if (check(TokenKind::Underscore)) {
        tree.setUnderscore();
        assert(eat(TokenKind::Underscore));
        // path as _
        // done
        return StringResult<ast::use_tree::UseTree>(tree);
      }
    } else if (!checkKeyWord(KeyWordKind::KW_AS)) {
      // path
      // done
      return StringResult<ast::use_tree::UseTree>(tree);
    }
  }
  return StringResult<ast::use_tree::UseTree>("failed to parse use tree");
}

} // namespace rust_compiler::parser
