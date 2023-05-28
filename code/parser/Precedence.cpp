#include "Parser/Precedence.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

Precedence Parser::getLeftBindingPower() {
  switch (getToken().getKind()) {
  case TokenKind::PathSep:
    return Precedence::Path;
  case TokenKind::Plus:
    return Precedence::Plus;
  case TokenKind::Minus:
    return Precedence::Minus;
  case TokenKind::Lt:
    return Precedence::LessThan;
  case TokenKind::Eq:
    return Precedence::Equal;
  case TokenKind::DotDot:
    return Precedence::DotDot;
  case TokenKind::DotDotEq:
    return Precedence::DotDotEq;
  case TokenKind::ParenOpen:
    return Precedence::FunctionCall;
  case TokenKind::PlusEq:
    return Precedence::PlusAssign;
  case TokenKind::SquareOpen:
    return Precedence::ArrayIndexing;
  case TokenKind::QMark:
    return Precedence::QuestionMark;
  case TokenKind::Dot: {
    if (getToken(1).isIdentifier() &&
        getToken(2).getKind() == TokenKind::ParenOpen)
      return Precedence::FieldExpression;
    return Precedence::MethodCall;
  }
  case TokenKind::Keyword: {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_AS: {
      return Precedence::As;
    }
    default: {
      llvm::errs() << KeyWord2String(getToken().getKeyWordKind()) << "\n";
      assert(false);
    }
    }
    break;
  }
  default: {
    llvm::errs() << "getLeftBindingPower: unknown token: "
                 << Token2String(getToken().getKind()) << "\n";
    return Precedence::Lowest;
    exit(EXIT_FAILURE);
  }
  }
}

} // namespace rust_compiler::parser
