#include "Parser/Parser.h"

#include "AST/GenericParam.h"
#include "AST/Module.h"
#include "AST/Visiblity.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <vector>

using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

bool Parser::checkIdentifier() { return check(TokenKind::Identifier); }

bool Parser::eatPathIdentSegment() {
  if (check(TokenKind::Identifier))
    return eat(TokenKind::Identifier);
  if (checkKeyWord(KeyWordKind::KW_SUPER))
    return eatKeyWord((KeyWordKind::KW_SUPER));
  if (checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return eatKeyWord((KeyWordKind::KW_SELFVALUE));
  if (checkKeyWord(KeyWordKind::KW_SELFTYPE))
    return eatKeyWord((KeyWordKind::KW_SELFTYPE));
  if (checkKeyWord(KeyWordKind::KW_CRATE))
    return eatKeyWord((KeyWordKind::KW_CRATE));
  if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE))
    return eatKeyWord((KeyWordKind::KW_DOLLARCRATE));
  return false;
}

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

bool Parser::checkLiteral() {
  return check(TokenKind::CHAR_LITERAL) || check(TokenKind::STRING_LITERAL) ||
         check(TokenKind::RAW_STRING_LITERAL) ||
         check(TokenKind::BYTE_LITERAL) ||
         check(TokenKind::BYTE_STRING_LITERAL) ||
         check(TokenKind::RAW_BYTE_STRING_LITERAL) ||
         check(TokenKind::INTEGER_LITERAL) || check(TokenKind::FLOAT_LITERAL) ||
         checkKeyWord(KeyWordKind::KW_TRUE) ||
         checkKeyWord(KeyWordKind::KW_FALSE);
}

CheckPoint Parser::getCheckPoint() { return CheckPoint(offset); }

void Parser::recover(const CheckPoint &cp) { offset = cp.readOffset(); }

bool Parser::checkOuterAttribute(uint8_t offset) {
  if (check(TokenKind::Hash, offset) &&
      check(TokenKind::SquareOpen, offset + 1))
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

  bool unsafe = false;

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    unsafe = true;
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  }

  if (checkKeyWord(lexer::KeyWordKind::KW_MOD) &&
      check(lexer::TokenKind::Identifier, 1) &&
      check(lexer::TokenKind::Semi, 2)) {
    // mod foo;
    assert(eatKeyWord(lexer::KeyWordKind::KW_MOD));
    Token token = getToken();
    std::string modName = token.getIdentifier();
    assert(eat(lexer::TokenKind::Identifier));
    assert(eat(lexer::TokenKind::Semi));

    Module mod = {loc, vis, ast::ModuleKind::Module, modName};
    if (unsafe)
      mod.setUnsafe();

    return std::make_shared<ast::Module>(mod);
  }

  if (checkKeyWord(lexer::KeyWordKind::KW_MOD) &&
      check(lexer::TokenKind::Identifier, 1) &&
      check(lexer::TokenKind::BraceOpen, 2)) {
    assert(eatKeyWord(lexer::KeyWordKind::KW_MOD));
    Token token = getToken();
    std::string modName = token.getIdentifier();
    assert(eat(lexer::TokenKind::Identifier));
    assert(eat(lexer::TokenKind::BraceOpen));
    // mod foo {}
    Module mod = {loc, vis, ast::ModuleKind::ModuleTree, modName};
    if (unsafe)
      mod.setUnsafe();

    if (checkInnerAttribute()) {
      llvm::Expected<std::vector<ast::InnerAttribute>> inner =
          parseInnerAttributes();
      if (auto e = inner.takeError()) {
        llvm::errs() << "failed to parse inner attributes in mod: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      mod.setInnerAttributes(*inner);
    }

    if (check(TokenKind::BraceClose)) {
      assert(eat(lexer::TokenKind::BraceClose));
      return std::make_shared<Module>(mod);
    }

    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse module: eof");
      } else if (check(TokenKind::BraceClose)) {
        // done
        assert(eat(lexer::TokenKind::BraceClose));
        return std::make_shared<Module>(mod);
      } else {
        llvm::Expected<std::shared_ptr<ast::Item>> item = parseItem();
        if (auto e = item.takeError()) {
          llvm::errs() << "failed to parse item in mod: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        mod.addItem(*item);
      }
    }
  }

  return createStringError(inconvertibleErrorCode(), "failed to parse module");
}

llvm::Expected<std::shared_ptr<ast::Crate>>
Parser::parseCrateModule(std::string_view crateName, basic::CrateNum crateNum) {
  Location loc = getLocation();

  Crate crate = {crateName, crateNum};

  if (checkInnerAttribute()) {

    llvm::Expected<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (auto e = innerAttributes.takeError()) {
      llvm::errs() << "failed to parse inner attributes in crate: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    crate.setInnerAttributes(*innerAttributes);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // done
      return std::make_shared<Crate>(crate);
    }
    llvm::Expected<std::shared_ptr<ast::Item>> item = parseItem();
    if (auto e = item.takeError()) {
      llvm::errs() << "failed to parse item in crate: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    crate.addItem(*item);
  }

  return std::make_shared<Crate>(crate);
}

llvm::Expected<std::shared_ptr<ast::WhereClauseItem>>
Parser::parseLifetimeWhereClauseItem() {
  Location loc = getLocation();
  LifetimeWhereClauseItem item = {loc};

  llvm::Expected<ast::Lifetime> lifetime = parseLifetimeAsLifetime();
  if (auto e = lifetime.takeError()) {
    llvm::errs() << "failed to parse lifetime in LifetimeWhereClauseItem: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  item.setForLifetimes(*lifetime);

  if (!check(TokenKind::Colon)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse :token in lifetime where clause item");
  }
  assert(eat(TokenKind::Colon));

  llvm::Expected<ast::LifetimeBounds> bounds = parseLifetimeBounds();
  if (auto e = bounds.takeError()) {
    llvm::errs()
        << "failed to parse lifetime bounds in LifetimeWhereClauseItem: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  item.setLifetimeBounds(*bounds);

  return std::make_shared<LifetimeWhereClauseItem>(item);
}

llvm::Expected<std::shared_ptr<ast::WhereClauseItem>>
Parser::parseTypeBoundWhereClauseItem() {
  Location loc = getLocation();

  TypeBoundWhereClauseItem item = {loc};

  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ast::types::ForLifetimes> forLifetime = parseForLifetimes();
    if (auto e = forLifetime.takeError()) {
      llvm::errs()
          << "failed to parse ForLifetime in TypeBoundWhereClauseItem: "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setForLifetimes(*forLifetime);
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in TypeBoundWhereClauseItem: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  item.setType(*type);

  if (!check(TokenKind::Colon))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse : token in TypeBoundWhereClauseItem");

  assert(eat(TokenKind::Colon));

  llvm::Expected<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
  if (auto e = bounds.takeError()) {
    llvm::errs()
        << "failed to parse type param bounds in TypeBoundWhereClauseItem: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  item.setBounds(*bounds);

  return std::make_shared<TypeBoundWhereClauseItem>(item);
}

llvm::Expected<std::shared_ptr<ast::WhereClauseItem>>
Parser::parseWhereClauseItem() {
  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseTypeBoundWhereClauseItem();
  if (checkLifetime())
    return parseLifetimeWhereClauseItem();
  return parseTypeBoundWhereClauseItem();
}

llvm::Expected<ast::WhereClause> Parser::parseWhereClause() {
  Location loc = getLocation();

  WhereClause where{loc};

  if (!checkKeyWord(KeyWordKind::KW_WHERE)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse where keyword in where clause");
  }

  assert(eatKeyWord(KeyWordKind::KW_WHERE));

  bool hasTrailingComma = false;

  while (checkWhereClauseItem()) {
    llvm::Expected<std::shared_ptr<ast::WhereClauseItem>> clauseItem =
        parseWhereClauseItem();
    if (auto e = clauseItem.takeError()) {
      llvm::errs() << "failed to parse where clause item in where clause: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    where.addWhereClauseItem(*clauseItem);
    hasTrailingComma = false;

    if (check(TokenKind::Comma)) {
      assert(check(TokenKind::Comma));
      hasTrailingComma = true;
    }
  }

  if (hasTrailingComma)
    where.setTrailingComma();

  return where;
}

llvm::Expected<ast::ConstParam> Parser::parseConstParam() {
  Location loc = getLocation();

  ConstParam param = {loc};

  if (!checkKeyWord(KeyWordKind::KW_CONST)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse const keyword in const param");

    assert(eatKeyWord(KeyWordKind::KW_CONST));
  }

  if (check(TokenKind::Identifier)) {
    Token tok = getToken();
    param.setIdentifier(tok.getIdentifier());
    assert(eat(TokenKind::Identifier));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in const param");
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in const param: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.setType(*type);

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));

    if (check(TokenKind::BraceOpen)) {
      llvm::Expected<std::shared_ptr<ast::Expression>> block =
          parseBlockExpression();
      if (auto e = type.takeError()) {
        llvm::errs() << "failed to parse block expression in const param: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      param.setBlock(*block);
    } else if (check(TokenKind::Identifier)) {
      Token tok = getToken();
      param.setInit(tok.getIdentifier());
      assert(check(TokenKind::Identifier));
    } else if (check(TokenKind::Minus)) {
      assert(check(TokenKind::Minus));
      llvm::Expected<std::shared_ptr<ast::Expression>> literal =
          parseLiteralExpression();
      if (auto e = literal.takeError()) {
        llvm::errs() << "failed to parse literal expression in const param: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      param.setInitLiteral(*literal);
    } else {
      llvm::Expected<std::shared_ptr<ast::Expression>> literal =
          parseLiteralExpression();
      if (auto e = literal.takeError()) {
        llvm::errs() << "failed to parse literal expression in const param: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      param.setInitLiteral(*literal);
    }
  }

  return param;
}

llvm::Expected<ast::LifetimeBounds> Parser::parseLifetimeBounds() {
  Location loc = getLocation();

  LifetimeBounds bounds = {loc};

  if (!checkLifetime())
    return bounds;

  bool trailingPlus = false;
  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (!checkLifetime()) {
      llvm::Expected<ast::Lifetime> life = parseLifetimeAsLifetime();
      if (auto e = life.takeError()) {
        llvm::errs() << "failed to parse life time in life time bounds: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      trailingPlus = false;
      bounds.setLifetime(*life);
      if (check(TokenKind::Plus)) {
        trailingPlus = true;
        assert(eat(TokenKind::Plus));
      }
    }
  }

  if (trailingPlus)
    bounds.setTrailingPlus();

  return bounds;
}

llvm::Expected<ast::LifetimeParam> Parser::parseLifetimeParam() {
  Location loc = getLocation();

  LifetimeParam param = {loc};

  llvm::Expected<ast::Lifetime> lifeTime = parseLifetimeAsLifetime();
  if (auto e = lifeTime.takeError()) {
    llvm::errs() << "failed to parse Lifetime in lifetime param: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.setLifetime(*lifeTime);

  if (!check(TokenKind::Colon))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse : token in lifetime param");
  assert(eat(TokenKind::Colon));

  llvm::Expected<ast::LifetimeBounds> bounds = parseLifetimeBounds();
  if (auto e = bounds.takeError()) {
    llvm::errs() << "failed to parse LifetimeBounds in lifetime param: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.setBounds(*bounds);

  return param;
}

llvm::Expected<ast::TypeParam> Parser::parseTypeParam() {
  Location loc = getLocation();

  TypeParam param = {loc};

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in type param");

  Token tok = getToken();
  param.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Colon) && check(TokenKind::Eq, 1)) {
    assert(eat(TokenKind::Colon));
    assert(eat(TokenKind::Eq));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type in type param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setType(*type);
    return param;
  } else if (check(TokenKind::Eq)) {
    // type
    assert(eat(TokenKind::Eq));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type in type param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setType(*type);
    return param;
  } else if (check(TokenKind::Colon) && !check(TokenKind::Eq, 1)) {
    // type param bounds

    llvm::Expected<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (auto e = bounds.takeError()) {
      llvm::errs() << "failed to parse type param bounds in type param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setBounds(*bounds);
    if (check(TokenKind::Eq)) {
      assert(eat(TokenKind::Eq));
      llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
          parseType();
      if (auto e = type.takeError()) {
        llvm::errs() << "failed to parse type in type param: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      param.setType(*type);
      return param;
    } else {
      return param;
    }
  } else if (!check(TokenKind::Colon) && !check(TokenKind::Eq)) {
    return param;
  }
  return param;
}

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

  GenericParams params = {loc};

  if (check(TokenKind::Lt) && check(TokenKind::Gt, 1)) {
    assert(eat(TokenKind::Lt));
    assert(eat(TokenKind::Gt));
    // done
  }

  if (check(TokenKind::Lt)) {
    assert(eat(TokenKind::Lt));
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse generic params with eof");
      } else if (check(TokenKind::Gt)) {
        return params;
      } else if (check(TokenKind::Comma) && check(TokenKind::Gt, 1)) {
        // done trailingComma
        params.setTrailingComma();
        return params;
      } else if (check(TokenKind::Gt)) {
        // done
        return params;
      } else {
        llvm::Expected<ast::GenericParam> generic = parseGenericParam();
        if (auto e = generic.takeError()) {
          llvm::errs() << "failed to parse generic param in generic params: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        params.addGenericParam(*generic);
      }
    }
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse generic params");
  }

  return params;
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
