#include "Parser/Parser.h"

#include "AST/Module.h"
#include "AST/Visiblity.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <vector>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

bool Parser::check(lexer::TokenKind token) {
  return ts.getAt(offset).getKind() == token;
}

bool Parser::check(lexer::TokenKind token, size_t _offset) {
  return ts.getAt(offset + _offset).getKind() == token;
}

bool Parser::checkKeyWord(lexer::KeyWordKind keyword) {
  if (ts.getAt(offset).getKind() == TokenKind::Keyword)
    return ts.getAt(offset).getKeyWordKind() == keyword;
  return false;
}

bool Parser::eatKeyWord(lexer::KeyWordKind keyword) {
  if (checkKeyWord(keyword)) {
    ++offset;
    return true;
  }
  ++offset;
  return false;
}

bool Parser::eat(lexer::TokenKind token) {
  if (check(token)) {
    ++offset;
    return true;
  }
  ++offset;
  return false;
}

rust_compiler::Location Parser::getLocation() {
  return ts.getAt(offset).getLocation();
}

lexer::Token Parser::getToken() { return ts.getAt(offset); }

llvm::Expected<std::vector<std::shared_ptr<ast::Item>>> Parser::parseItems() {
  if (check(lexer::TokenKind::Hash) && check(lexer::TokenKind::Not, 1) &&
      check(lexer::TokenKind::SquareOpen, 2)) {
    llvm::Expected<std::vector<ast::OuterAttribute>> result =
        parseOuterAttributes();
  }
  //  llvm::Expected<ast::Item> item = parseItem();
  // FIXME
}

llvm::Expected<std::shared_ptr<ast::VisItem>> Parser::parseVisItem() {
  std::optional<ast::Visibility> vis;

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
      return parseFunction(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_UNSAFE, 1)) {
      return parseFunction(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_EXTERN, 1)) {
      return parseFunction(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_FN, 1)) {
      return parseFunction(vis);
    } else {
      return parseConstantItem(vis);
    }
  } else if (checkKeyWord(lexer::KeyWordKind::KW_ASYNC)) {
    return parseFunction(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_UNSAFE)) {
    if (checkKeyWord(lexer::KeyWordKind::KW_TRAIT, 1)) {
      return parseTrait(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_MOD, 1)) {
      return parseMod(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_IMPL, 1)) {
      return parseImplementation(vis);
    } else if (checkKeyWord(lexer::KeyWordKind::KW_EXTERN, 1)) {
      return parseExternBlock(vis);
    }

    return parseFunction(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_MOD)) {
    return parseMod(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_FN)) {
    return parseFunction(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_USE)) {
    return parseUseDeclaration(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_TYPE)) {
    return parseTypeAlias(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_STRUCT)) {
    return parseStruct(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_ENUM)) {
    return parseEnumeration(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_UNION)) {
    return parseUnion(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_STATIC)) {
    return parseStaticItem(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_TRAIT)) {
    return parseTrait(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_IMPL)) {
    return parseImplementation(vis);
  } else if (checkKeyWord(lexer::KeyWordKind::KW_EXTERN)) {
    return parseExternBlock(vis);
  }
  // complete?
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseMod(std::optional<ast::Visibility> vis) {

  Location loc = getLocation();

  if (checkKeyWord(lexer::KeyWordKind::KW_MOD) &&
      check(lexer::TokenKind::Identifier, 1) &&
      check(lexer::TokenKind::Semi, 2)) {
    // mod foo;
    assert(eatKeyWord(lexer::KeyWordKind::KW_MOD));
    Token token = getToken();
    std::string modName = token.getIdentifier();
    assert(eat(lexer::TokenKind::Identifier));
    assert(eat(lexer::TokenKind::Semi));

    return std::make_shared<ast::Module>(loc, vis, ast::ModuleKind::Module,
                                         modName);
  }

  if (checkKeyWord(lexer::KeyWordKind::KW_MOD) &&
      check(lexer::TokenKind::Identifier, 1) &&
      check(lexer::TokenKind::BraceOpen, 2)) {
    // mod foo {}
  }

  // error
}

llvm::Expected<std::shared_ptr<ast::Crate>>
Parser::parseCrateModule(std::string_view crateName) {
  assert(false);
  Location loc = getLocation();

  llvm::Expected<std::vector<ast::InnerAttribute>> innerAttributes =
      parseInnerAttributes();
  if (auto e = innerAttributes.takeError()) {
    llvm::errs() << "failed to parse inner attributes: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  // parseInnerAttributes

  llvm::Expected<std::vector<std::shared_ptr<ast::Item>>> items = parseItems();
  if (auto e = items.takeError()) {
    llvm::errs() << "failed to parse items: " << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  // parseItems
}

} // namespace rust_compiler::parser
