#include "AST/AssociatedItem.h"
#include "AST/ConstantItem.h"
#include "AST/ExternBlock.h"
#include "AST/Implementation.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructStruct.h"
#include "AST/TupleStruct.h"
#include "AST/Union.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "llvm/Support/Error.h"

#include <optional>

using namespace llvm;
using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<ast::AssociatedItem> Parser::parseAssociatedItem() {
  Location loc = getLocation();
  AssociatedItem item = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setOuterAttributes(*outer);
  }

  if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
    llvm::Expected<std::shared_ptr<ast::MacroItem>> macroItem =
        parseMacroInvocationSemiMacroItem();
    if (auto e = macroItem.takeError()) {
      llvm::errs() << "failed to parse outer attributes in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setMacroItem(*macroItem);
    return item;
  } else if (checkKeyWord(KeyWordKind::KW_PUB)) {
    llvm::Expected<ast::Visibility> vis = parseVisibility();
    if (auto e = vis.takeError()) {
      llvm::errs() << "failed to parse visiblity in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setVisiblity(*vis);
    if (checkKeyWord(KeyWordKind::KW_TYPE)) {
      // TypeAlias
      llvm::Expected<std::shared_ptr<ast::VisItem>> typeAlias =
          parseTypeAlias(*vis);
      if (auto e = typeAlias.takeError()) {
        llvm::errs() << "failed to parse type alias in associated item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      item.setTypeAlias(*typeAlias);
      return item;
    } else if (checkKeyWord(KeyWordKind::KW_CONST) &&
               check(TokenKind::Colon, 2)) {
      // ConstantItem
      llvm::Expected<std::shared_ptr<ast::VisItem>> constantItem =
          parseConstantItem(*vis);
      if (auto e = constantItem.takeError()) {
        llvm::errs() << "failed to parse constant item in associated item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      item.setConstantItem(*constantItem);
      return item;
    } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
               checkKeyWord(KeyWordKind::KW_ASYNC) ||
               checkKeyWord(KeyWordKind::KW_UNSAFE) ||
               checkKeyWord(KeyWordKind::KW_EXTERN) ||
               checkKeyWord(KeyWordKind::KW_FN)) {
      // fun
      llvm::Expected<std::shared_ptr<ast::VisItem>> fun = parseFunction(*vis);
      if (auto e = fun.takeError()) {
        llvm::errs() << "failed to parse function in associated item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      item.setFunction(*fun);
      return item;
    } else {
      // error
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse associated item");
    }
  } else if (checkKeyWord(KeyWordKind::KW_TYPE)) {
    // type alias
    llvm::Expected<std::shared_ptr<ast::VisItem>> typeAlias =
        parseTypeAlias(std::nullopt);
    if (auto e = typeAlias.takeError()) {
      llvm::errs() << "failed to parse type alias in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setTypeAlias(*typeAlias);
    return item;
  } else if (checkKeyWord(KeyWordKind::KW_CONST) &&
             check(TokenKind::Colon, 2)) {
    // constant item
    llvm::Expected<std::shared_ptr<ast::VisItem>> constantItem =
        parseConstantItem(std::nullopt);
    if (auto e = constantItem.takeError()) {
      llvm::errs() << "failed to parse constant item in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setConstantItem(*constantItem);
    return item;
  } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
             checkKeyWord(KeyWordKind::KW_ASYNC) ||
             checkKeyWord(KeyWordKind::KW_UNSAFE) ||
             checkKeyWord(KeyWordKind::KW_EXTERN) ||
             checkKeyWord(KeyWordKind::KW_FN)) {
    // fun
    llvm::Expected<std::shared_ptr<ast::VisItem>> fun =
        parseFunction(std::nullopt);
    if (auto e = fun.takeError()) {
      llvm::errs() << "failed to parse function in associated item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    item.setFunction(*fun);
    return item;
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse associated item");
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse associated item");
}

llvm::Expected<ast::ExternalItem> Parser::parseExternalItem() {
  Location loc = getLocation();

  ExternalItem impl = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in external item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setOuterAttributes(*outer);
  }

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    llvm::Expected<ast::Visibility> vis = parseVisibility();
    if (auto e = vis.takeError()) {
      llvm::errs() << "failed to parse visibility in external item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    if (checkKeyWord(KeyWordKind::KW_STATIC)) {
      llvm::Expected<std::shared_ptr<ast::VisItem>> stat =
          parseStaticItem(*vis);
      if (auto e = stat.takeError()) {
        llvm::errs() << "failed to parse static item in external item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      impl.setStaticItem(*stat);
      return impl;
    } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
               checkKeyWord(KeyWordKind::KW_ASYNC) ||
               checkKeyWord(KeyWordKind::KW_EXTERN) ||
               checkKeyWord(KeyWordKind::KW_FN) ||
               checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      llvm::Expected<std::shared_ptr<ast::VisItem>> fn = parseFunction(*vis);
      if (auto e = fn.takeError()) {
        llvm::errs() << "failed to parse function in external item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      impl.setFunction(*fn);
      return impl;
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse external item");
    }
  } else if (checkKeyWord(KeyWordKind::KW_STATIC)) {
    llvm::Expected<std::shared_ptr<ast::VisItem>> stat =
        parseStaticItem(std::nullopt);
    if (auto e = stat.takeError()) {
      llvm::errs() << "failed to parse static item in external item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setStaticItem(*stat);
    return impl;
  } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
             checkKeyWord(KeyWordKind::KW_ASYNC) ||
             checkKeyWord(KeyWordKind::KW_EXTERN) ||
             checkKeyWord(KeyWordKind::KW_FN) ||
             checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    llvm::Expected<std::shared_ptr<ast::VisItem>> fn =
        parseFunction(std::nullopt);
    if (auto e = fn.takeError()) {
      llvm::errs() << "failed to parse function in external item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setFunction(*fn);
    return impl;
  } else if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
    // Macro
    llvm::Expected<std::shared_ptr<ast::Expression>> macro =
        parseMacroInvocationExpression();
    if (auto e = macro.takeError()) {
      llvm::errs() << "failed to parse MacroInvocation in external item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setMacroInvocation(*macro);
    return impl;
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse external item");
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseExternBlock(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  ExternBlock impl = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
    impl.setUnsafe();
  }

  if (!checkKeyWord(KeyWordKind::KW_EXTERN))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse extern keyword in extern block");
  assert(eatKeyWord(KeyWordKind::KW_EXTERN));

  llvm::Expected<ast::Abi> abi = parseAbi();
  if (auto e = abi.takeError()) {
    llvm::errs() << "failed to parse abi in extern block : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  impl.setAbi(*abi);

  if (!check(TokenKind::BraceOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in extern block");
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (auto e = innerAttributes.takeError()) {
      llvm::errs() << "failed to parse inner attributes in extern block : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setInnerAttributes(*innerAttributes);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (check(TokenKind::BraceClose)) {
      // done
      assert(eat(TokenKind::BraceClose));
      return std::make_shared<ExternBlock>(impl);
    } else {
      llvm::Expected<ast::ExternalItem> item = parseExternalItem();
      if (auto e = item.takeError()) {
        llvm::errs() << "failed to parse external item in extern block : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      impl.addItem(*item);
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse  extern block");
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseImplementation(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    recover(cp);
    return parseTraitImpl(vis);
  }

  if (!checkKeyWord(KeyWordKind::KW_IMPL))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse impl keyword in implementation");

  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> generic = parseGenericParams();
    if (auto e = generic.takeError()) {
      llvm::errs() << "failed to parse  generic params in implementation : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (check(TokenKind::Not)) {
    recover(cp);
    return parseTraitImpl(vis);
  }

  llvm::Expected<std::shared_ptr<ast::types::TypePath>> path = parseTypePath();
  if (path) {
    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      recover(cp);
      return parseTraitImpl(vis);
    }
  }

  recover(cp);
  return parseInherentImpl(vis);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseTypeAlias(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  TypeAlias alias = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_TYPE))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse type keyword in type alias");

  assert(eatKeyWord(KeyWordKind::KW_TYPE));

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in type alias");

  Token id = getToken();
  alias.setIdentifier(id.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    if (auto e = genericParams.takeError()) {
      llvm::errs() << "failed to parse  generic params in type alias : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    alias.setGenericParams(*genericParams);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<TypeAlias>(alias);
  }

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    llvm::Expected<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (auto e = bounds.takeError()) {
      llvm::errs() << "failed to parse  type param bounds in type alias : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    alias.setParamBounds(*bounds);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<TypeAlias>(alias);
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in type alias : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    alias.setWhereClause(*where);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<TypeAlias>(alias);
  } else if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse where keyword or ; token in type alias");
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in type alias : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  alias.setType(*type);

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in type alias : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    alias.setTypeWhereClause(*where);
  } else if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<TypeAlias>(alias);
  }
  {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse where keyword or ; token in type alias");
  }

  assert(eat(TokenKind::Semi));
  return std::make_shared<TypeAlias>(alias);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStaticItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  StaticItem stat = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STATIC))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse static keyword in static item");

  assert(eatKeyWord(KeyWordKind::KW_STATIC));

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    stat.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  }

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in static item");

  Token id = getToken();
  stat.setIdentifier(id.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Semi))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse : in static item");
  assert(eat(TokenKind::Semi));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (auto e = typeExpr.takeError()) {
    llvm::errs() << "failed to parse type expression in constant item : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  stat.setType(*typeExpr);

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<StaticItem>(stat);
  } else if (check(TokenKind::Eq)) {
    // initializer
    llvm::Expected<std::shared_ptr<ast::Expression>> init = parseExpression();
    if (auto e = init.takeError()) {
      llvm::errs() << "failed to parse  expression in constant item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    stat.setInit(*init);
    assert(eat(TokenKind::Semi));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse static item");
  }
  return std::make_shared<StaticItem>(stat);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseConstantItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  ConstantItem con = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_CONST))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse const keyword in constant item");

  assert(eatKeyWord(KeyWordKind::KW_CONST));

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    con.setIdentifier("_");
  } else if (check(TokenKind::Identifier)) {
    Token id = getToken();
    con.setIdentifier(id.getIdentifier());
    assert(eat(TokenKind::Identifier));
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse identifier token in constant item");
  }

  if (!check(TokenKind::Colon)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse colon token in constant item");
  }

  assert(eat(TokenKind::Colon));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (auto e = typeExpr.takeError()) {
    llvm::errs() << "failed to parse type expression in constant item : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  con.setType(*typeExpr);

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<ConstantItem>(con);
  } else if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    // initializer
    llvm::Expected<std::shared_ptr<ast::Expression>> init = parseExpression();
    if (auto e = init.takeError()) {
      llvm::errs() << "failed to parse  expression in constant item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    con.setInit(*init);
    assert(eat(TokenKind::Semi));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse constant item");
  }

  return std::make_shared<ConstantItem>(con);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseUnion(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Union uni = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_UNION))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse union keyword in union");

  assert(eatKeyWord(KeyWordKind::KW_UNION));

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in union");

  assert(check(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    if (auto e = genericParams.takeError()) {
      llvm::errs() << "failed to parse generic params in union : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    uni.setGenericParams(*genericParams);
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClause = parseWhereClause();
    if (auto e = whereClause.takeError()) {
      llvm::errs() << "failed to parse  where clause in union : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    uni.setWhereClause(*whereClause);
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in union");
  }
  assert(check(TokenKind::BraceOpen));

  llvm::Expected<ast::StructFields> fields = parseStructFields();
  if (auto e = fields.takeError()) {
    llvm::errs() << "failed to parse  struct fields in union : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  uni.setStructfields(*fields);
  assert(check(TokenKind::BraceClose));

  return std::make_shared<Union>(uni);
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStruct(std::optional<ast::Visibility> vis) {
  CheckPoint cp = getCheckPoint();

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct keyword in struct");

  assert(checkKeyWord(KeyWordKind::KW_STRUCT));

  if (!check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in struct");
  }
  assert(check(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    if (auto e = genericParams.takeError()) {
      llvm::errs() << "failed to parse generic params in struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClause = parseWhereClause();
    if (auto e = whereClause.takeError()) {
      llvm::errs() << "failed to parse where clause in struct : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    recover(cp);
    return parseStructStruct(vis);
  }

  if (check(TokenKind::Semi)) {
    recover(cp);
    return parseStructStruct(vis);
  }

  if (check(TokenKind::BraceOpen)) {
    recover(cp);
    return parseStructStruct(vis);
  }

  recover(cp);
  return parseTupleStruct(vis);
}

} // namespace rust_compiler::parser
