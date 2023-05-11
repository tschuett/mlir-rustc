#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTypeNoBounds() {

//  llvm::errs() << "parseTypeNoBounds: " << Token2String(getToken().getKind())
//               << "\n";

  if (checkKeyWord(KeyWordKind::KW_IMPL))
    return parseImplType();

  if (check(TokenKind::Not))
    return parseNeverType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  if (check(TokenKind::Star))
    return parseRawPointerType();

  if (check(TokenKind::Underscore))
    return parseInferredType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

  if (check(TokenKind::And))
    return parseReferenceType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_EXTERN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FN))
    return parseBareFunctionType();

  if (check(TokenKind::Lt))
    return parseQualifiedPathInType();

  return parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType();
}

} // namespace rust_compiler::parser

// KW_FOR -> BareFunctionType ?

/*
  TraitObjectType
  TypePath
  MacroInvocation
 */

/*
  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseBareFunctionType();
    TraitBound
*/
