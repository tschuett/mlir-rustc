#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::types;

namespace rust_compiler::parser {

PathKind Parser::testTypePathOrSimplePath() {
  if (check(TokenKind::PathSep)) {
    assert(eat(TokenKind::PathSep));
  }

  while (true) {
    if (check(TokenKind::Identifier) || checkKeyWord(KeyWordKind::KW_SUPER) ||
        checkKeyWord(KeyWordKind::KW_SELFVALUE) ||
        checkKeyWord(KeyWordKind::KW_CRATE) ||
        checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
      assert(eat());
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SELFTYPE)) {
      return PathKind::TypePath;
    } else if (check(TokenKind::Lt)) {
      // GenericArgs
      return PathKind::TypePath;
    } else if (check(TokenKind::ParenOpen)) {
      // TypePathFn
      return PathKind::TypePath;
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      continue;
    } else if (check(TokenKind::Eof)) {
      return PathKind::Unknown;
    } else {
      return PathKind::SimplePath;
    }
  }
  // SimplePath?
}
// TypePath: PathIdentSegment
// SimplePath: SimplePathSegment

// eof

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
    parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType() {
  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ForLifetimes> forLifetimes = parseForLifetimes();
    // check error

    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      // BareFunctionType
    } else if (check(TokenKind::PathSep)) {
      // TraitObjectType
    }
  } else { // not for
    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      // BareFunctionType
    } else if (check(TokenKind::PathSep)) {
      // TraitObjectType: TypePath
      // TypePath: TypePath
      // MacroInvocation: SimplePath
    }
    // BareFunctionType or TraitObjectType
  }
  // else if (check(TokenKind::ParenOpen)) {
  //   // ParenType or TupleType or TraitObjectType
  //   //    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> types =
  //   //       parseType();
  // }
  // else if (check(TokenKind::PathSeq)) {
  //   // SimplePath in MacroInvocation
  //   // TypePath
  //   // Or TraitObjectType
  // }
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseType() {
  if (check(TokenKind::Star))
    return parseRawPointerType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  if (checkKeyWord(KeyWordKind::KW_IMPL))
    return parseImplType();

  if (check(TokenKind::Not))
    return parseNeverType();

  if (check(TokenKind::And))
    return parseReferenceType();

  if (check(TokenKind::Underscore))
    return parseInferredType();

  if (check(TokenKind::Lt))
    return parseQualifiedPathInType();

  if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1))
    return parseTupleType();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_EXTERN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

  if (check(TokenKind::QMark))
    return parseTraitObjectType();
}

} // namespace rust_compiler::parser

/*
  TypePath or MacroInvocation
  TraitObjectType
  BareFunctionType with ForLifetimes
 */
