#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::parseType() {
  if (check(TokenKind::ParenOpen))
    return parseTupleOrParensType();

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

  if (checkKeyWord(KeyWordKind::KW_UNSAFE))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_EXTERN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

}

} // namespace rust_compiler::parser

/*
  TypePath or MacroInvocation
  TraitObjectType
 */
