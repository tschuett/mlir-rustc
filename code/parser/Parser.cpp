#include "Parser/Parser.h"

#include "ADT/Result.h"
#include "AST/GenericParam.h"
#include "AST/InnerAttribute.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "AST/Visiblity.h"
#include "AST/WhereClause.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <llvm/Support/raw_ostream.h>
#include <string>
#include <vector>

using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::parser {

bool Parser::checkPostFix() {
  return check(TokenKind::QMark) || check(TokenKind::Plus) ||
         check(TokenKind::Minus) || check(TokenKind::Star) ||
         check(TokenKind::Slash) || check(TokenKind::Percent) ||
         check(TokenKind::Or) || check(TokenKind::Caret) ||
         check(TokenKind::Shl) || check(TokenKind::Shr) ||
         check(TokenKind::EqEq) || check(TokenKind::Ne) ||
         check(TokenKind::Gt) || check(TokenKind::Lt) || check(TokenKind::Ge) ||
         check(TokenKind::Le) || check(TokenKind::OrOr) ||
         check(TokenKind::AndAnd) || checkKeyWord(KeyWordKind::KW_AS) ||
         check(TokenKind::PlusEq) || check(TokenKind::MinusEq) ||
         check(TokenKind::StarEq) || check(TokenKind::SlashEq) ||
         check(TokenKind::PercentEq) || check(TokenKind::AndEq) ||
         check(TokenKind::AndEq) || check(TokenKind::OrEq) ||
         check(TokenKind::ShlEq) || check(TokenKind::Dot) ||
         check(TokenKind::SquareOpen) || check(TokenKind::DotDot) ||
         check(TokenKind::DotDotEq) || check(TokenKind::ShrEq);
}

bool Parser::checkMacroItem() {
  if (checkKeyWord(KeyWordKind::KW_MACRO_RULES))
    return true;

  if (checkSimplePathSegment())
    return true;

  return false;
}

bool Parser::checkStaticOrUnderscore() {
  if (checkKeyWord(KeyWordKind::KW_STATICLIFETIME))
    return true;

  if (check(TokenKind::LIFETIME_TOKEN) && getToken().getStorage() == "'_")
    return true;
  return false;
}

bool Parser::checkLifetime(uint8_t offset) {
  if (check(TokenKind::LIFETIME_OR_LABEL, offset)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_STATICLIFETIME, offset)) {
    return true;
  } else if (check(TokenKind::LIFETIME_TOKEN, offset) &&
             getToken(offset).getStorage() == "'_") {
    return true;
  }
  return false;
}

StringResult<ast::Lifetime> Parser::parseLifetimeAsLifetime() {
  Location loc = getLocation();
  Lifetime lf = {loc};

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
    lf.setLifetime(getToken().getStorage());
  } else if (checkKeyWord(KeyWordKind::KW_STATICLIFETIME)) {
    lf.setLifetime(getToken().getStorage());
  } else if (check(TokenKind::LIFETIME_TOKEN) &&
             getToken().getStorage() == "'_") {
    lf.setLifetime(getToken().getStorage());
  } else {
    return StringResult<ast::Lifetime>("failed to parse lifetime");
  }

  return StringResult<ast::Lifetime>(lf);
}

bool Parser::checkDelimiters() {
  if (check(TokenKind::BraceOpen)) {
    return true;
  } else if (check(TokenKind::BraceClose)) {
    return true;
  } else if (check(TokenKind::SquareOpen)) {
    return true;
  } else if (check(TokenKind::SquareClose)) {
    return true;
  } else if (check(TokenKind::ParenOpen)) {
    return true;
  } else if (check(TokenKind::ParenClose)) {
    return true;
  }

  return false;
}

bool Parser::eatSimplePathSegment() {
  if (check(TokenKind::Identifier))
    return eat(TokenKind::Identifier);
  if (checkKeyWord(KeyWordKind::KW_SUPER))
    return eatKeyWord(KeyWordKind::KW_SUPER);
  if (checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return eatKeyWord(KeyWordKind::KW_SELFVALUE);
  if (checkKeyWord(KeyWordKind::KW_SELFTYPE))
    return eatKeyWord(KeyWordKind::KW_SELFTYPE);
  if (checkKeyWord(KeyWordKind::KW_CRATE))
    return eatKeyWord(KeyWordKind::KW_CRATE);
  if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE))
    return eatKeyWord(KeyWordKind::KW_DOLLARCRATE);
  return false;
}

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
bool Parser::checkPathIdentSegment(uint8_t off) {
  if (check(TokenKind::Identifier, off))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SUPER, off))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SELFVALUE, off))
    return true;
  if (checkKeyWord(KeyWordKind::KW_SELFTYPE, off))
    return true;
  if (checkKeyWord(KeyWordKind::KW_CRATE, off))
    return true;
  if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE, off))
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

bool Parser::checkLiteral(uint8_t off) {
  return check(TokenKind::CHAR_LITERAL, off) ||
         check(TokenKind::STRING_LITERAL, off) ||
         check(TokenKind::RAW_STRING_LITERAL, off) ||
         check(TokenKind::BYTE_LITERAL, off) ||
         check(TokenKind::BYTE_STRING_LITERAL, off) ||
         check(TokenKind::RAW_BYTE_STRING_LITERAL, off) ||
         check(TokenKind::INTEGER_LITERAL, off) ||
         check(TokenKind::FLOAT_LITERAL, off) ||
         checkKeyWord(KeyWordKind::KW_TRUE, off) ||
         checkKeyWord(KeyWordKind::KW_FALSE, off);
}

CheckPoint Parser::getCheckPoint() { return CheckPoint(offset); }

void Parser::recover(const CheckPoint &cp) { offset = cp.readOffset(); }

bool Parser::checkOuterAttribute(uint8_t off) {
  if (check(TokenKind::Hash, off) && check(TokenKind::SquareOpen, off + 1))
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

lexer::Token Parser::getToken(uint8_t off) { return ts.getAt(offset + off); }

StringResult<std::shared_ptr<ast::VisItem>>
Parser::parseVisItem(std::span<OuterAttribute>) {
  std::optional<ast::Visibility> vis;

  if (checkKeyWord(lexer::KeyWordKind::KW_PUB)) {
    // FIXME
    Result<ast::Visibility, std::string> result = parseVisibility();
    if (!result) {
      llvm::errs() << "failed to parse visibility: " << result.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    vis = result.getValue();
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
  return StringResult<std::shared_ptr<ast::VisItem>>(
      "failed to parse vis item");
}

StringResult<std::shared_ptr<ast::VisItem>>
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

    return StringResult<std::shared_ptr<ast::VisItem>>(
        StringResult<std::shared_ptr<ast::VisItem>>(
            std::make_shared<ast::Module>(mod)));
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
      StringResult<std::vector<ast::InnerAttribute>> inner =
          parseInnerAttributes();
      if (!inner) {
        llvm::errs() << "failed to parse inner attributes in mod: "
                     << inner.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      std::vector<InnerAttribute> in = inner.getValue();
      mod.setInnerAttributes(in);
    }

    if (check(TokenKind::BraceClose)) {
      assert(eat(lexer::TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::VisItem>>(
          std::make_shared<Module>(mod));
    }

    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::VisItem>>(
            "failed to parse module: eof");
      } else if (check(TokenKind::BraceClose)) {
        // done
        assert(eat(lexer::TokenKind::BraceClose));
        return StringResult<std::shared_ptr<ast::VisItem>>(
            std::make_shared<Module>(mod));
      } else {
        Result<std::shared_ptr<ast::Item>, std::string> item = parseItem();
        if (!item) {
          llvm::errs() << "failed to parse item in mod: " << item.getError()
                       << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        mod.addItem(item.getValue());
      }
    }
  }

  return StringResult<std::shared_ptr<ast::VisItem>>("failed to parse module");
}

Result<std::shared_ptr<ast::Crate>, std::string>
Parser::parseCrateModule(std::string_view crateName, basic::CrateNum crateNum) {
  Location loc = getLocation();

  Crate crate = {crateName, crateNum};

  if (checkInnerAttribute()) {

    StringResult<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (!innerAttributes) {
      llvm::errs() << "failed to parse inner attributes in crate : "
                   << innerAttributes.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<InnerAttribute> inner = innerAttributes.getValue();
    crate.setInnerAttributes(inner);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // done
      return Result<std::shared_ptr<ast::Crate>, std::string>(
          std::make_shared<Crate>(crate));
    }
    Result<std::shared_ptr<ast::Item>, std::string> item = parseItem();
    if (!item) {
      llvm::errs() << "failed to parse item in crate: " << item.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    crate.addItem(item.getValue());
  }

  return Result<std::shared_ptr<ast::Crate>, std::string>(
      std::make_shared<Crate>(crate));
}

Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
Parser::parseLifetimeWhereClauseItem() {
  Location loc = getLocation();
  LifetimeWhereClauseItem item = {loc};

  StringResult<ast::Lifetime> lifetime = parseLifetimeAsLifetime();
  if (!lifetime) {
    llvm::errs() << "failed to parse lifetime in where clause item: "
                 << lifetime.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  item.setForLifetimes(lifetime.getValue());

  if (!check(TokenKind::Colon)) {
    return Result<std::shared_ptr<ast::WhereClauseItem>, std::string>(
        "failed to parse :token in lifetime where clause item");
  }
  assert(eat(TokenKind::Colon));

  Result<ast::LifetimeBounds, std::string> bounds = parseLifetimeBounds();
  if (!bounds) {
    llvm::errs() << "failed to parse lifetime bounds in where clause item: "
                 << bounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  item.setLifetimeBounds(bounds.getValue());

  return Result<std::shared_ptr<ast::WhereClauseItem>, std::string>(
      std::make_shared<LifetimeWhereClauseItem>(item));
}

Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
Parser::parseTypeBoundWhereClauseItem() {
  Location loc = getLocation();

  TypeBoundWhereClauseItem item = {loc};

  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    StringResult<ast::types::ForLifetimes> forLifetime = parseForLifetimes();
    if (!forLifetime) {
      llvm::errs()
          << "failed to parse lifetime in type bound where clause item: "
          << forLifetime.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setForLifetimes(forLifetime.getValue());
  }

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in TypeBoundWhereClauseItem: "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  item.setType(type.getValue());

  if (!check(TokenKind::Colon))
    return Result<std::shared_ptr<ast::WhereClauseItem>, std::string>(
        "failed to parse : token in TypeBoundWhereClauseItem");

  assert(eat(TokenKind::Colon));

  Result<ast::types::TypeParamBounds, std::string> bounds =
      parseTypeParamBounds();
  if (!bounds) {
    llvm::errs()
        << "failed to parse type param bounds in type bound where clause item: "
        << bounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  item.setBounds(bounds.getValue());

  return Result<std::shared_ptr<ast::WhereClauseItem>, std::string>(
      std::make_shared<TypeBoundWhereClauseItem>(item));
}

Result<std::shared_ptr<ast::WhereClauseItem>, std::string>
Parser::parseWhereClauseItem() {
  if (checkKeyWord(KeyWordKind::KW_FOR))
    return parseTypeBoundWhereClauseItem();
  if (checkLifetime())
    return parseLifetimeWhereClauseItem();
  return parseTypeBoundWhereClauseItem();
}

Result<ast::WhereClause, std::string> Parser::parseWhereClause() {
  Location loc = getLocation();

  WhereClause where{loc};

  if (!checkKeyWord(KeyWordKind::KW_WHERE)) {
    return Result<ast::WhereClause, std::string>(
        "failed to parse where keyword in where clause");
  }

  assert(eatKeyWord(KeyWordKind::KW_WHERE));

  while (true) {
    if (check(TokenKind::Eof)) {
      return Result<ast::WhereClause, std::string>(
          "failed to parse where clause: eof");
    } else if (check(TokenKind::BraceOpen)) {
      return Result<ast::WhereClause, std::string>(where);
    } else if (check(TokenKind::Semi)) {
      return Result<ast::WhereClause, std::string>(where);
    } else if (check(TokenKind::Comma) && check(TokenKind::BraceOpen, 1)) {
      assert(eat(TokenKind::Comma));
      where.setTrailingComma();
      return Result<ast::WhereClause, std::string>(where);
    } else if (check(TokenKind::Comma) && check(TokenKind::Semi, 1)) {
      assert(eat(TokenKind::Comma));
      where.setTrailingComma();
      return Result<ast::WhereClause, std::string>(where);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
    } else {
      Result<std::shared_ptr<ast::WhereClauseItem>, std::string> clauseItem =
          parseWhereClauseItem();
      if (!clauseItem) {
        llvm::errs() << "failed to parse where clause item in where clause: "
                     << clauseItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      where.addWhereClauseItem(clauseItem.getValue());
    }
  }
  return Result<ast::WhereClause, std::string>("failed to parse where clause");
}

Result<ast::ConstParam, std::string> Parser::parseConstParam() {
  Location loc = getLocation();

  ConstParam param = {loc};

  if (!checkKeyWord(KeyWordKind::KW_CONST)) {
    return Result<ast::ConstParam, std::string>(
        "failed to parse const keyword in const param");

    assert(eatKeyWord(KeyWordKind::KW_CONST));
  }

  if (check(TokenKind::Identifier)) {
    Token tok = getToken();
    param.setIdentifier(tok.getIdentifier());
    assert(eat(TokenKind::Identifier));
  } else {
    return Result<ast::ConstParam, std::string>(
        "failed to parse identifier token in const param");
  }

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in const param: " << type.getError()
                 << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.setType(type.getValue());

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));

    if (check(TokenKind::BraceOpen)) {
      Result<std::shared_ptr<ast::Expression>, std::string> block =
          parseBlockExpression({});
      if (!block) {
        llvm::errs() << "failed to parse block expression in const param: "
                     << block.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      param.setBlock(block.getValue());
    } else if (check(TokenKind::Identifier)) {
      Token tok = getToken();
      param.setInit(tok.getIdentifier());
      assert(check(TokenKind::Identifier));
    } else if (check(TokenKind::Minus)) {
      assert(check(TokenKind::Minus));
      Result<std::shared_ptr<ast::Expression>, std::string> literal =
          parseLiteralExpression({});
      if (!literal) {
        llvm::errs() << "failed to parse literal expression in const param: "
                     << literal.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      param.setInitLiteral(literal.getValue());
    } else {
      Result<std::shared_ptr<ast::Expression>, std::string> literal =
          parseLiteralExpression({});
      if (!literal) {
        llvm::errs() << "failed to parse literal expression in const param: "
                     << literal.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      param.setInitLiteral(literal.getValue());
    }
  }

  return Result<ast::ConstParam, std::string>(param);
}

StringResult<ast::LifetimeBounds> Parser::parseLifetimeBounds() {
  Location loc = getLocation();

  LifetimeBounds bounds = {loc};

  if (!checkLifetime())
    return StringResult<ast::LifetimeBounds>(bounds);

  bool trailingPlus = false;
  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<ast::LifetimeBounds>(
          "failed to pars elifetime bounds");
    } else if (!checkLifetime()) {
      StringResult<ast::Lifetime> life = parseLifetimeAsLifetime();
      if (!life) {
        llvm::errs() << "failed to parse lifetime in lifetime bounds: "
                     << life.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      trailingPlus = false;
      bounds.setLifetime(life.getValue());
      if (check(TokenKind::Plus)) {
        trailingPlus = true;
        assert(eat(TokenKind::Plus));
      }
    }
  }

  if (trailingPlus)
    bounds.setTrailingPlus();

  return StringResult<ast::LifetimeBounds>(bounds);
}

StringResult<ast::LifetimeParam> Parser::parseLifetimeParam() {
  Location loc = getLocation();

  LifetimeParam param = {loc};

  StringResult<ast::Lifetime> lifeTime = parseLifetimeAsLifetime();
  if (!lifeTime) {
    llvm::errs() << "failed to parse lifetime in lifetime param: "
                 << lifeTime.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.setLifetime(lifeTime.getValue());

  if (!check(TokenKind::Colon))
    return StringResult<ast::LifetimeParam>(
        "failed to parse : token in lifetime param");
  assert(eat(TokenKind::Colon));

  StringResult<ast::LifetimeBounds> bounds = parseLifetimeBounds();
  if (!bounds) {
    llvm::errs() << "failed to parse lifetime bounds in lifetime param: "
                 << bounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.setBounds(bounds.getValue());

  return StringResult<ast::LifetimeParam>(param);
}

StringResult<ast::TypeParam> Parser::parseTypeParam() {
  Location loc = getLocation();

  TypeParam param = {loc};

  if (!check(TokenKind::Identifier))
    return StringResult<ast::TypeParam>(
        "failed to parse identifier token in type param");

  Token tok = getToken();
  param.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Colon) && check(TokenKind::Eq, 1)) {
    assert(eat(TokenKind::Colon));
    assert(eat(TokenKind::Eq));
    Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to parse type in type param: " << type.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setType(type.getValue());
    return StringResult<ast::TypeParam>(param);
  } else if (check(TokenKind::Eq)) {
    // type
    assert(eat(TokenKind::Eq));
    Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to parse type in type param: " << type.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setType(type.getValue());
    return StringResult<ast::TypeParam>(param);
  } else if (check(TokenKind::Colon) && !check(TokenKind::Eq, 1)) {
    // type param bounds

    StringResult<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (!bounds) {
      llvm::errs() << "failed to parse type param bounds in type param: "
                   << bounds.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setBounds(bounds.getValue());
    if (check(TokenKind::Eq)) {
      assert(eat(TokenKind::Eq));
      Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
          parseType();
      if (!type) {
        llvm::errs() << "failed to parse type in type param: "
                     << type.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      param.setType(type.getValue());
      return StringResult<ast::TypeParam>(param);
    } else {
      return StringResult<ast::TypeParam>(param);
    }
  } else if (!check(TokenKind::Colon) && !check(TokenKind::Eq)) {
    return StringResult<ast::TypeParam>(param);
  }
  return StringResult<ast::TypeParam>(param);
}

StringResult<ast::GenericParam> Parser::parseGenericParam() {
  Location loc = getLocation();

  GenericParam param = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in generic param: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<OuterAttribute> out = outer.getValue();
    param.setOuterAttributes(out);
  }

  if (checkKeyWord(KeyWordKind::KW_CONST)) {
    StringResult<ast::ConstParam> constParam = parseConstParam();
    if (!constParam) {
      llvm::errs() << "failed to parse const param in generic param: "
                   << constParam.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setConstParam(constParam.getValue());
  } else if (check(TokenKind::LIFETIME_OR_LABEL)) {
    StringResult<ast::LifetimeParam> lifetimeParam = parseLifetimeParam();
    if (!lifetimeParam) {
      llvm::errs() << "failed to parse life time param in generic param: "
                   << lifetimeParam.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setLifetimeParam(lifetimeParam.getValue());
  } else if (check(TokenKind::Identifier)) {
    StringResult<ast::TypeParam> typeParam = parseTypeParam();
    if (!typeParam) {
      llvm::errs() << "failed to parse type param in generic param: "
                   << typeParam.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setTypeParam(typeParam.getValue());
  } else {
    // report
    return StringResult<ast::GenericParam>("failed to parse generic param");
  }

  return StringResult<ast::GenericParam>(param);
}

StringResult<ast::GenericParams> Parser::parseGenericParams() {
  Location loc = getLocation();

  GenericParams params = {loc};

  if (check(TokenKind::Lt) && check(TokenKind::Gt, 1)) {
    assert(eat(TokenKind::Lt));
    assert(eat(TokenKind::Gt));
    // done
    return StringResult<ast::GenericParams>(params);
  }

  if (check(TokenKind::Lt)) {
    assert(eat(TokenKind::Lt));
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<ast::GenericParams>(
            "failed to parse generic params with eof");
      } else if (check(TokenKind::Gt)) {
        assert(eat(TokenKind::Gt));
        return StringResult<ast::GenericParams>(params);
      } else if (check(TokenKind::Comma) && check(TokenKind::Gt, 1)) {
        // done trailingComma
        assert(eat(TokenKind::Comma));
        assert(eat(TokenKind::Gt));
        params.setTrailingComma();
        return StringResult<ast::GenericParams>(params);
      } else {
        StringResult<ast::GenericParam> generic = parseGenericParam();
        if (!generic) {
          llvm::errs() << "failed to parse generic param in generic params: "
                       << generic.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        params.addGenericParam(generic.getValue());
      }
    }
  } else {
    return StringResult<ast::GenericParams>("failed to parse generic params");
  }

  return StringResult<ast::GenericParams>(params);
}

Result<ast::Visibility, std::string> Parser::parseVisibility() {
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
    return Result<ast::Visibility, std::string>(vis);
  } else if (checkKeyWord(KeyWordKind::KW_PUB) &&
             check(TokenKind::ParenOpen, 1) &&
             checkKeyWord(KeyWordKind::KW_SELFVALUE, 2) &&
             check(TokenKind::ParenClose, 3)) {
    // pub (self)
    vis.setKind(VisibilityKind::PublicSelf);
    return Result<ast::Visibility, std::string>(vis);
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
    return Result<ast::Visibility, std::string>(vis);
  } else if (checkKeyWord(KeyWordKind::KW_PUB) &&
             check(TokenKind::ParenOpen, 1) &&
             checkKeyWord(KeyWordKind::KW_IN, 2)) {
    // pub (in ...)
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    assert(eat(TokenKind::ParenOpen));
    assert(eatKeyWord(KeyWordKind::KW_IN));
    StringResult<ast::SimplePath> simple = parseSimplePath();
    if (!simple) {
      llvm::errs() << "failed to parse simple path in visibility: "
                   << simple.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    vis.setPath(simple.getValue());
    if (!check(TokenKind::ParenClose))
      // report error
      assert(eat(TokenKind::ParenClose));
    // done
    vis.setKind(VisibilityKind::PublicIn);
    return Result<ast::Visibility, std::string>(vis);
  } else if (checkKeyWord(KeyWordKind::KW_PUB)) {
    // pub
    assert(eatKeyWord(KeyWordKind::KW_PUB));
    vis.setKind(VisibilityKind::Public);
    return Result<ast::Visibility, std::string>(vis);
  }
  // private
  vis.setKind(VisibilityKind::Private);
  return Result<ast::Visibility, std::string>(vis);
}

} // namespace rust_compiler::parser
