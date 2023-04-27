#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkPathOrStructOrMacro() {
  llvm::errs() << "checkPathOrStructOrMacro"
               << "\n";

  if (check(TokenKind::Semi))
    return false;

  if (check(TokenKind::Eof))
    return false;

  CheckPoint cp = getCheckPoint();
  while (true) {
    llvm::errs() << "checkPathOrStructOrMacro: "
                 << Token2String(getToken().getKind()) << "\n";
    if (getToken().isKeyWord())
      llvm::errs() << KeyWord2String(getToken().getKeyWordKind()) << "\n";
    if (check(TokenKind::Eof)) {
      recover(cp);
      return false;
    } else if (check(TokenKind::Not)) {
      recover(cp);
      return true;
    } else if (checkKeyWord(KeyWordKind::KW_STRUCT)) {
      recover(cp);
      return true;
    } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      recover(cp);
      return false;
    } else if (checkKeyWord(KeyWordKind::KW_TYPE)) {
      recover(cp);
      return false;
    } else if (check(TokenKind::PathSep)) {
      assert((eat(TokenKind::PathSep)));
    } else if (check(TokenKind::BraceOpen)) {
      recover(cp);
      return true;
    } else if (check(TokenKind::Lt)) {
      recover(cp);
      return true;
    } else if (checkSimplePathSegment()) {
      assert(eatSimplePathSegment());
    } else if (checkPathIdentSegment()) {
      assert(eatPathIdentSegment());
    }
  }
  return false;
}

bool Parser::checkExpressionWithoutBlock() {
//  llvm::errs() << "checkExpressionWithoutBlock"
//               << "\n";
  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in check expression "
                      "without block: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
  }

  if (check(TokenKind::And) || check(TokenKind::AndAnd)) {
    return true;
  } else if (checkLiteral()) {
    return true;
  } else if (check(TokenKind::Star)) {
    return true;
  } else if (check(TokenKind::Lt)) {
    return true;
  } else if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    return true;
  } else if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    return true;
  } else if (check(TokenKind::PathSep) || checkPathIdentSegment()) {
    return true;
  } else if (check(TokenKind::PathSep) || checkPathIdentSegment()) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_MOVE) || check(TokenKind::Or, 1)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_MOVE) || check(TokenKind::OrOr, 1)) {
    return true;
  } else if (check(TokenKind::Or)) {
    return true;
  } else if (check(TokenKind::OrOr)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_RETURN)) {
    return true;
  } else if (check(TokenKind::Underscore)) {
    return true;
  } else if (check(TokenKind::ParenOpen)) {
    return true;
  } else if (check(TokenKind::SquareOpen)) {
    return true;
  } else if (check(TokenKind::DotDot)) {
    return true;
  } else if (check(TokenKind::DotDotEq)) {
    return true;
  }

  // path or macro invocation or struct

  if (checkPathOrStructOrMacro())
    return true;

  //llvm::errs() << Token2String(getToken().getKind()) << "\n";

  Restrictions restritions;
  StringResult<std::shared_ptr<ast::Expression>> left =
      parseExpression({}, restritions);
  if (!left) {
    llvm::errs() << "failed to parse expression in check expression "
                    "without block: "
                 << left.getError() << "\n";
    printFunctionStack();
  }

  return checkExpressionWithoutBlock(left.getValue());
}

bool Parser::checkExpressionWithoutBlock(std::shared_ptr<Expression> lhs) {
  if (check(TokenKind::QMark)) {
    return true;
  } else if (check(TokenKind::QMark)) {
    return true;
  } else if (check(TokenKind::Plus)) {
    return true;
  } else if (check(TokenKind::Minus)) {
    return true;
  } else if (check(TokenKind::Star)) {
    return true;
  } else if (check(TokenKind::Slash)) {
    return true;
  } else if (check(TokenKind::Percent)) {
    return true;
  } else if (check(TokenKind::And)) {
    return true;
  } else if (check(TokenKind::Or)) {
    return true;
  } else if (check(TokenKind::Caret)) {
    return true;
  } else if (check(TokenKind::Shl)) {
    return true;
  } else if (check(TokenKind::Shr)) {
    return true;
  } else if (check(TokenKind::EqEq)) {
    return true;
  } else if (check(TokenKind::Ne)) {
    return true;
  } else if (check(TokenKind::Gt)) {
    return true;
  } else if (check(TokenKind::Lt)) {
    return true;
  } else if (check(TokenKind::Ge)) {
    return true;
  } else if (check(TokenKind::Le)) {
    return true;
  } else if (check(TokenKind::OrOr)) {
    return true;
  } else if (check(TokenKind::AndAnd)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_AS)) {
    return true;
  } else if (check(TokenKind::Eq)) {
    return true;
  } else if (check(TokenKind::PlusEq)) {
    return true;
  } else if (check(TokenKind::MinusEq)) {
    return true;
  } else if (check(TokenKind::StarEq)) {
    return true;
  } else if (check(TokenKind::SlashEq)) {
    return true;
  } else if (check(TokenKind::PercentEq)) {
    return true;
  } else if (check(TokenKind::CaretEq)) {
    return true;
  } else if (check(TokenKind::OrEq)) {
    return true;
  } else if (check(TokenKind::ShlEq)) {
    return true;
  } else if (check(TokenKind::ShrEq)) {
    return true;
  } else if (check(TokenKind::Dot) && checkKeyWord(KeyWordKind::KW_AWAIT, 1)) {
    return true;
  } else if (check(TokenKind::SquareOpen)) {
    return true;
  } else if (check(TokenKind::Dot) && check(TokenKind::INTEGER_LITERAL, 1)) {
    return true;
  } else if (check(TokenKind::ParenOpen)) {
    return true;
  } else if (check(TokenKind::Dot) && checkPathExprSegment(1)) {
    return true;
  } else if (check(TokenKind::Dot) && check(TokenKind::Identifier, 1)) {
    return true;
  } else if (check(TokenKind::DotDot)) {
    return true;
  } else if (check(TokenKind::DotDotEq)) {
    return true;
  }

  return false;
}

} // namespace rust_compiler::parser
