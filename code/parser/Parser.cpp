#include "Parser/Parser.h"

#include "AST/Module.h"
#include "AST/Visiblity.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <vector>

using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

/// IDENTIFIER | super | self | Self | crate | $crate
bool Parser::checkPathIdentSegment() {
  if (check(TokenKind::Identifier))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SUPER))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SELFTYPE))
    return true;
  if (checkKeyWord(KeyWordKind::KW_CRATE))
    return true;
  if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE))
    return true;
  return false;
}

/// IDENTIFIER | super | self | crate | $crate
bool Parser::checkSimplePathSegment() {
  if (check(TokenKind::Identifier))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SUPER))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return true;
  if (checkKeyWord(KeyWordKind::KW_CRATE))
    return true;
  if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE))
    return true;
  return false;
}

bool Parser::checkOuterAttribute() {
  if (check(TokenKind::Hash) && check(TokenKind::SquareOpen, 1))
    return true;
  return false;
}

bool Parser::checkInnerAttribute() {
  if (check(TokenKind::Hash) && check(TokenKind::Not, 1) &&
      check(TokenKind::SquareOpen, 2))
    return true;
  return false;
}

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

bool Parser::checkKeyWord(lexer::KeyWordKind keyword, size_t _offset) {
  if (ts.getAt(offset + _offset).getKind() == TokenKind::Keyword)
    return ts.getAt(offset + _offset).getKeyWordKind() == keyword;
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
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse vis item");
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

llvm::Expected<ast::WhereClause> Parser::parseWhereClause() {
  Location loc = getLocation();

  WhereClause where{loc};

  if (!checkKeyWord(KeyWordKind::KW_WHERE)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse where keyword in where clause");
  }

  // FIXME
}

llvm::Expected<ast::ConstParam> Parser::parseConstParam() {}

llvm::Expected<ast::LifetimeParam> Parser::parseLifetimeParam() {}

llvm::Expected<ast::TypeParam> Parser::parseTypeParam() {}

llvm::Expected<ast::GenericParam> Parser::parseGenericParam() {
  Location loc = getLocation();

  GenericParam param = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in generic param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setOuterAttributes(*outer);
  }

  if (checkKeyWord(KeyWordKind::KW_CONST)) {
    llvm::Expected<ast::ConstParam> constParam = parseConstParam();
    if (auto e = constParam.takeError()) {
      llvm::errs() << "failed to parse const param in generic param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setConstParam(*constParam);
  } else if (check(TokenKind::LIFETIME_OR_LABEL)) {
    llvm::Expected<ast::LifetimeParam> lifetimeParam = parseLifetimeParam();
    if (auto e = lifetimeParam.takeError()) {
      llvm::errs() << "failed to parse lifetime param in generic param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setLifetimeParam(*lifetimeParam);
  } else if (check(TokenKind::Identifier)) {
    llvm::Expected<ast::TypeParam> typeParam = parseTypeParam();
    if (auto e = typeParam.takeError()) {
      llvm::errs() << "failed to parse type param in generic param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setTypeParam(*typeParam);
  } else {
    // report
  }

  return param;
}

llvm::Expected<ast::GenericParams> Parser::parseGenericParams() {
  Location loc = getLocation();

  if (check(TokenKind::Lt) && check(TokenKind::Gt, 1)) {
    // done
  }

  if (check(TokenKind::Lt)) {
    parseGenericParam();
  }

  // FIXME
}

llvm::Expected<ast::Visibility> Parser::parseVisibility() {
  Location loc = getLocation();
  Visibility vis = {loc};

  if (checkKeyWord(KeyWordKind::KW_PUB) && check(TokenKind::ParenOpen, 1) &&
      checkKeyWord(KeyWordKind::KW_CRATE, 2) &&
      check(TokenKind::ParenClose, 3)) {
    // pub (crate)
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    assert(eat(TokenKind::ParenOpen));
    assert(eatKeyWord(KeyWordKind::KW_CRATE));
    assert(eat(TokenKind::ParenClose));
    vis.setKind(VisibilityKind::PublicCrate);
    return vis;
  } else if (checkKeyWord(KeyWordKind::KW_PUB) &&
             check(TokenKind::ParenOpen, 1) &&
             checkKeyWord(KeyWordKind::KW_SELFVALUE, 2) &&
             check(TokenKind::ParenClose, 3)) {
    // pub (self)
    vis.setKind(VisibilityKind::PublicSelf);
    return vis;
  } else if (checkKeyWord(KeyWordKind::KW_PUB) &&
             check(TokenKind::ParenOpen, 1) &&
             checkKeyWord(KeyWordKind::KW_SUPER, 2) &&
             check(TokenKind::ParenClose, 3)) {
    // pub (super)
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    assert(eat(TokenKind::ParenOpen));
    assert(eatKeyWord(KeyWordKind::KW_SUPER));
    assert(eat(TokenKind::ParenClose));
    vis.setKind(VisibilityKind::PublicSuper);
    return vis;
  } else if (checkKeyWord(KeyWordKind::KW_PUB) &&
             check(TokenKind::ParenOpen, 1) &&
             checkKeyWord(KeyWordKind::KW_IN, 2)) {
    // pub (in ...)
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    assert(eat(TokenKind::ParenOpen));
    assert(eatKeyWord(KeyWordKind::KW_IN));
    llvm::Expected<ast::SimplePath> simple = parseSimplePath();
    if (auto e = simple.takeError()) {
      llvm::errs() << "failed to parse simple path in visibility: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    vis.setPath(*simple);
    if (!check(TokenKind::ParenClose))
      // report error
      assert(eat(TokenKind::ParenClose));
    // done
    vis.setKind(VisibilityKind::PublicIn);
    return vis;
  } else if (checkKeyWord(KeyWordKind::KW_PUB)) {
    // pub
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    vis.setKind(VisibilityKind::Public);
    return vis;
  }
  // private
  vis.setKind(VisibilityKind::Private);
  return vis;
}

} // namespace rust_compiler::parser
