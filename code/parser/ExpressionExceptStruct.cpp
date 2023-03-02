#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionExceptStruct() {
  // copy and paste
  CheckPoint cp = getCheckPoint();

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs()
          << "failed to parse outer attributes in expression except struct: "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_LOOP)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_MATCH)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_IF)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (check(TokenKind::BraceOpen)) {
    recover(cp);
    return parseExpressionWithBlock();
  } else if (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon)) {
    recover(cp);
    return parseExpressionWithBlock();
  }

  recover(cp);
  return parseExpressionWithoutBlockExceptStruct();
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseExpressionWithoutBlockExceptStruct() {

  // copy and paste

  std::vector<ast::OuterAttribute> attributes;
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outerAttributes =
        parseOuterAttributes();
    if (auto e = outerAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    attributes = *outerAttributes;
  }

  // FIXME attributes

  if (checkLiteral()) {
    return parseLiteralExpression();
  }
  if (check(TokenKind::And)) {
    return parseBorrowExpression();
  }

  if (check(TokenKind::Lt)) {
    return parseQualifiedPathInExpression();
  }

  if (check(TokenKind::AndAnd)) {
    return parseBorrowExpression();
  }

  if (check(TokenKind::Star)) {
    return parseDereferenceExpression();
  }

  if (check(TokenKind::Not) || check(TokenKind::Minus)) {
    return parseNegationExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::OrOr)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::Or)) {
    return parseClosureExpression();
  }

  if (check(TokenKind::SquareOpen)) {
    return parseArrayExpression();
  }

  if (check(TokenKind::ParenOpen)) {
    return parseGroupedOrTupleExpression();
  }

  if (check(TokenKind::DotDot)) {
    return parseRangeExpression();
  }

  if (check(TokenKind::DotDotEq)) {
    return parseRangeExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    return parseAsyncBlockExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_CONTINUE)) {
    return parseContinueExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_BREAK)) {
    return parseBreakExpression();
  }

  if (checkKeyWord(KeyWordKind::KW_RETURN)) {
    return parseReturnExpression();
  }

  if (check(TokenKind::Underscore)) {
    return parseUnderScoreExpression();
  }

  if (check(TokenKind::PathSep)) {
    /*
      PathInExpression -> PathExpression, : forrbidden StructExprStruct,
      StructExprTuple, StructExprUnit SimplePath -> MacroInvocation
     */
    return parsePathInExpressionOrMacroInvocationExpression();
  }

  if (check(TokenKind::Identifier) || checkKeyWord(KeyWordKind::KW_SUPER) ||
      checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
      checkKeyWord(KeyWordKind::KW_CRATE) ||
      checkKeyWord(KeyWordKind::KW_SELFTYPE) ||
      checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
    return parsePathInExpressionOrMacroInvocationExpression();
  }

  //checkPostFix()


  return parseExpressionWithPostfix();
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parsePathInExpressionOrMacroInvocationExpression() {
  // copy and paste

  CheckPoint cp = getCheckPoint();

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse PathInExpressionOrMacroInvocationExpression: eof");
    } else if (checkPathIdentSegment()) {
      assert(eatPathIdentSegment());
    } else if (checkSimplePathSegment()) {
      assert(eatSimplePathSegment());
    } else if (check(TokenKind::Not)) {
      recover(cp);
      return parseMacroInvocationExpression();
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
    } else if (check(TokenKind::BraceOpen)) {
      // terminator: never a struct
      recover(cp);
      return parsePathInExpression();
    } else if (check(TokenKind::Lt)) {
      // generics
      recover(cp);
      return parsePathInExpression();
    }
  }
}

// FIXME : generics

} // namespace rust_compiler::parser
