#include "Parser/Parser2.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <vector>

namespace rust_compiler::parser {

bool Parser::checkInKeyWords(std::span<lexer::KeyWordKind> keywords) {}

llvm::Expected<std::vector<std::shared_ptr<ast::Item>>> Parser::parseItems() {
  //  llvm::Expected<ast::Item> item = parseItem();
  // FIXME
}

llvm::Expected<std::shared_ptr<ast::VisItem>> Parser::parseVisItem() {
  ast::Visibility vis;

  if (checkKeyWord(lexer::KeyWordKind::KW_PUB)) {
    // FIXME
    llvm::Expected<ast::Visibility> result = parseVisibility();
    if (auto e = result.takeError()) {
      llvm::errs() << "failed to parse visiblity: " << toString(std::move(e))
                   << "\n";
      exit(EXIT_FAILURE);
    }
    vis = *result;
  }

  if (checkKeyWord(lexer::KeyWordKind::KW_CONST)) {
    if (checkKeyWord(lexer::KeyWordKind::KW_ASYNC, 1)) {
      // fun
    } else if (checkKeyWord(lexer::KeyWordKind::KW_UNSAFE, 1)) {
      // fun
    } else if (checkKeyWord(lexer::KeyWordKind::KW_EXTERN, 1)) {
      // fun
    } else if (checkKeyWord(lexer::KeyWordKind::KW_FN, 1)) {
      // fun
    } else {
      // constant item
    }
    // fun or constant item
  } else if (checkKeyWord(lexer::KeyWordKind::KW_ASYNC)) {
    // fun
  } else if (checkKeyWord(lexer::KeyWordKind::KW_UNSAFE)) {
    // fun or trait or mod or traitimpl or extern block
  } else if (checkKeyWord(lexer::KeyWordKind::KW_FN)) {
    return parseFunction(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_USE)) {
    //
    return parseUseDeclaration();
  } else if (checkKeyWord(lexer::KeyWordKind::KW_TYPE)) {
    return parseTypeAlias();
  } else if (checkKeyWord(lexer::KeyWordKind::KW_STRUCT)) {
  } else if (checkKeyWord(lexer::KeyWordKind::KW_ENUM)) {
    return parseEnumeration();
  } else if (checkKeyWord(lexer::KeyWordKind::KW_UNION)) {
    return parseUnion();
  } else if (checkKeyWord(lexer::KeyWordKind::KW_STATIC)) {
    return parseStaticItem();
  } else if (checkKeyWord(lexer::KeyWordKind::KW_TRAIT)) {
    return parseTrait();
  }
}

llvm::Expected<ast::Attribute> Parser::parseAttribute() {}

} // namespace rust_compiler::parser
