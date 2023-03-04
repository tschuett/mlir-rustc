#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

/*
  TraitObjectType: TypePath followed by ???
  TypePath: TypePath
  MacroInvocation: SimplePath followed by !
 */

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectTypeOrTypePathOrMacroInvocation() {
  CheckPoint cp = getCheckPoint();

  while (true) {
    //llvm::outs() << lexer::Token2String(getToken().getKind()) << "\n";
    if (check(TokenKind::Eof)) {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse "
          "raitObjectTypeOrTypePathOrMacroInvocation: eof");
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
    } else if (check(TokenKind::QMark)) {
      recover(cp);
      return parseTraitObjectType();
    } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
      recover(cp);
      return parseTraitObjectType();
    } else if (check(TokenKind::Not)) {
      recover(cp);
      return parseMacroInvocationType();
    } else if (check(TokenKind::ParenClose)) {
      recover(cp);
      return parseTypePath();
    } else if (check(TokenKind::Lt)) {
      recover(cp);
      return parseTypePath();
    } else if (check(TokenKind::Plus)) {
      recover(cp);
      return parseTraitObjectType();
    } else if (checkSimplePathSegment()) {
      assert(eatSimplePathSegment());
    } else if (checkLifetime()) {
      recover(cp);
      return parseTraitObjectType();
    } else if (checkPathIdentSegment()) {
      assert(eatPathIdentSegment());
    } else if (check(TokenKind::BraceOpen)) {
      // terminator
      recover(cp);
      return parseTypePath();
    } else if (check(TokenKind::Comma)) {
      // terminator
      recover(cp);
      return parseTypePath();
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse "
                           "raitObjectTypeOrTypePathOrMacroInvocation");
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
    parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType() {
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();

  //llvm::outs() << "parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObje"
  //                "ctTypeOrBareFunctionType"
  //             << "\n";

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    return parseTraitObjectType();
  } else if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1)) {
    return parseTupleType();
  } else if (check(TokenKind::ParenOpen)) {
    return parseTupleOrParensType();
  } else if (check(TokenKind::QMark)) {
    return parseTraitObjectType();
  } else if (checkKeyWord(KeyWordKind::KW_FN)) {
    return parseBareFunctionType();
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    return parseBareFunctionType();
  } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    return parseBareFunctionType();
  } else if (check(TokenKind::PathSep)) {
    return parseTraitObjectTypeOrTypePathOrMacroInvocation();
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
    if (auto e = forL.takeError()) {
      llvm::errs() << "failed to parse for lifetimes in  : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      recover(cp);
      return parseBareFunctionType();
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      recover(cp);
      return parseBareFunctionType();
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      recover(cp);
      return parseBareFunctionType();
    }
    recover(cp);
    return parseTraitObjectType();
  } else if (checkLifetime()) {
    recover(cp);
    return parseTraitObjectType();
  } else {
    return parseTraitObjectTypeOrTypePathOrMacroInvocation();
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse "
                           "TupleOrParensTypeOrTypePathOrMacroInvocationOrTrait"
                           "ObjectTypeOrBareFunctionType");
}

} // namespace rust_compiler::parser