#include "AST/AssociatedItem.h"
#include "AST/ConstantItem.h"
#include "AST/ExternBlock.h"
#include "AST/Implementation.h"
#include "AST/MacroInvocationSemiItem.h"
#include "AST/MacroInvocationSemiStatement.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructStruct.h"
#include "AST/TupleStruct.h"
#include "AST/Union.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "llvm/Support/CommandLine.h"

#include <llvm/Support/Error.h>
#include <optional>

using namespace llvm;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Item>>
Parser::parseMacroItem(std::span<OuterAttribute>) {
  if (checkKeyWord(KeyWordKind::KW_MACRO_RULES))
    return parseMacroRulesDefinition();

  return parseMacroInvocationSemiItem();
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseMacroInvocationSemiItem() {
  Location loc = getLocation();
  MacroInvocationSemiItem macro = {loc};

  StringResult<ast::SimplePath> path = parseSimplePath();
  if (!path) {
    llvm::errs() << "failed to parse simple path in macro invocation item: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setPath(path.getValue());

  if (!check(TokenKind::Not)) {
    llvm::errs()
        << "failed to parse ! token in macro invocation semi statement: "
        << "\n";
    exit(EXIT_FAILURE);
  }
  assert(eat(TokenKind::Not));

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse macro invocation semi statement: eof");
    } else if (check(TokenKind::ParenOpen)) {
      macro.setKind(MacroInvocationSemiItemKind::Paren);
      assert(eat(TokenKind::ParenOpen));
    } else if (check(TokenKind::SquareOpen)) {
      macro.setKind(MacroInvocationSemiItemKind::Square);
      assert(eat(TokenKind::SquareOpen));
    } else if (check(TokenKind::BraceOpen)) {
      macro.setKind(MacroInvocationSemiItemKind::Brace);
      assert(eat(TokenKind::BraceOpen));
    } else if (check(TokenKind::ParenClose) && check(TokenKind::Semi, 1)) {
      if (macro.getKind() != MacroInvocationSemiItemKind::Paren)
        return StringResult<std::shared_ptr<ast::Item>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::ParenClose));
      assert(eat(TokenKind::Semi));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<MacroInvocationSemiItem>(macro));
    } else if (check(TokenKind::SquareClose) && check(TokenKind::Semi, 1)) {
      if (macro.getKind() != MacroInvocationSemiItemKind::Square)
        return StringResult<std::shared_ptr<ast::Item>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::SquareClose));
      assert(eat(TokenKind::Semi));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<MacroInvocationSemiItem>(macro));
    } else if (check(TokenKind::BraceClose)) {
      if (macro.getKind() != MacroInvocationSemiItemKind::Brace)
        return StringResult<std::shared_ptr<ast::Item>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<MacroInvocationSemiItem>(macro));
    } else {
      StringResult<ast::TokenTree> tree = parseTokenTree();
      if (!tree) {
        llvm::errs() << "failed to parse token tree in macro invocation item: "
                     << tree.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      macro.addTree(tree.getValue());
    }
  }

  return StringResult<std::shared_ptr<ast::Item>>(
      "failed to parse macro invocation semi statement");
}

StringResult<ast::AssociatedItem> Parser::parseAssociatedItem() {
  Location loc = getLocation();
  AssociatedItem item = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in associated item: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> out = outer.getValue();
    item.setOuterAttributes(out);
  }

  if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
    StringResult<std::shared_ptr<ast::Item>> macroItem =
        parseMacroInvocationSemiItem();
    if (!macroItem) {
      llvm::errs()
          << "failed to parse macro invocation semi in associated item: "
          << macroItem.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setMacroItem(macroItem.getValue());
    return StringResult<ast::AssociatedItem>(item);
  } else if (checkKeyWord(KeyWordKind::KW_PUB)) {
    StringResult<ast::Visibility> vis = parseVisibility();
    if (!vis) {
      llvm::errs() << "failed to parse visibility in associated item: "
                   << vis.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setVisiblity(vis.getValue());
    if (checkKeyWord(KeyWordKind::KW_TYPE)) {
      // TypeAlias
      StringResult<std::shared_ptr<ast::Item>> typeAlias =
          parseTypeAlias(vis.getValue());
      if (!typeAlias) {
        llvm::errs() << "failed to parse type alias in associated item: "
                     << typeAlias.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      item.setTypeAlias(typeAlias.getValue());
      return StringResult<ast::AssociatedItem>(item);
    } else if (checkKeyWord(KeyWordKind::KW_CONST) &&
               check(TokenKind::Colon, 2)) {
      // ConstantItem
      StringResult<std::shared_ptr<ast::Item>> constantItem =
          parseConstantItem(vis.getValue());
      if (!constantItem) {
        llvm::errs() << "failed to parse constant item in associated item: "
                     << constantItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      item.setConstantItem(constantItem.getValue());
      return StringResult<ast::AssociatedItem>(item);
    } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
               checkKeyWord(KeyWordKind::KW_ASYNC) ||
               checkKeyWord(KeyWordKind::KW_UNSAFE) ||
               checkKeyWord(KeyWordKind::KW_EXTERN) ||
               checkKeyWord(KeyWordKind::KW_FN)) {
      // fun
      StringResult<std::shared_ptr<ast::Item>> fun =
          parseFunction(vis.getValue());
      if (!fun) {
        llvm::errs() << "failed to parse function in associated item: "
                     << fun.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      item.setFunction(fun.getValue());
      return StringResult<ast::AssociatedItem>(item);
    } else {
      // error
      return StringResult<ast::AssociatedItem>(
          "failed to parse associated item");
    }
  } else if (checkKeyWord(KeyWordKind::KW_TYPE)) {
    // type alias
    StringResult<std::shared_ptr<ast::Item>> typeAlias =
        parseTypeAlias(std::nullopt);
    if (!typeAlias) {
      llvm::errs() << "failed to parse type alias in associated item: "
                   << typeAlias.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setTypeAlias(typeAlias.getValue());
    return StringResult<ast::AssociatedItem>(item);
  } else if (checkKeyWord(KeyWordKind::KW_CONST) &&
             check(TokenKind::Colon, 2)) {
    // constant item
    StringResult<std::shared_ptr<ast::Item>> constantItem =
        parseConstantItem(std::nullopt);
    if (!constantItem) {
      llvm::errs() << "failed to parse constant item in associated item: "
                   << constantItem.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setConstantItem(constantItem.getValue());
    return StringResult<ast::AssociatedItem>(item);
  } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
             checkKeyWord(KeyWordKind::KW_ASYNC) ||
             checkKeyWord(KeyWordKind::KW_UNSAFE) ||
             checkKeyWord(KeyWordKind::KW_EXTERN) ||
             checkKeyWord(KeyWordKind::KW_FN)) {
    // fun
    StringResult<std::shared_ptr<ast::Item>> fun =
        parseFunction(std::nullopt);
    if (!fun) {
      llvm::errs() << "failed to parse function in associated item: "
                   << fun.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    item.setFunction(fun.getValue());
    return StringResult<ast::AssociatedItem>(item);
  } else {
    return StringResult<ast::AssociatedItem>("failed to parse associated item");
  }
  return StringResult<ast::AssociatedItem>("failed to parse associated item");
}

StringResult<ast::ExternalItem> Parser::parseExternalItem() {
  Location loc = getLocation();

  ExternalItem impl = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in external item: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> out = outer.getValue();
    impl.setOuterAttributes(out);
  }

  if (checkKeyWord(KeyWordKind::KW_PUB)) {
    StringResult<ast::Visibility> vis = parseVisibility();
    if (!vis) {
      llvm::errs() << "failed to parse visibility in external item: "
                   << vis.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    if (checkKeyWord(KeyWordKind::KW_STATIC)) {
      StringResult<std::shared_ptr<ast::Item>> stat =
          parseStaticItem(vis.getValue());
      if (!stat) {
        llvm::errs() << "failed to parse static item in external item: "
                     << stat.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      impl.setStaticItem(stat.getValue());
      return StringResult<ast::ExternalItem>(impl);
    } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
               checkKeyWord(KeyWordKind::KW_ASYNC) ||
               checkKeyWord(KeyWordKind::KW_EXTERN) ||
               checkKeyWord(KeyWordKind::KW_FN) ||
               checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      StringResult<std::shared_ptr<ast::Item>> fn =
          parseFunction(vis.getValue());
      if (!fn) {
        llvm::errs() << "failed to parse function in external item: "
                     << fn.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      impl.setFunction(fn.getValue());
      return StringResult<ast::ExternalItem>(impl);
    } else {
      return StringResult<ast::ExternalItem>("failed to parse external item");
    }
  } else if (checkKeyWord(KeyWordKind::KW_STATIC)) {
    StringResult<std::shared_ptr<ast::Item>> stat =
        parseStaticItem(std::nullopt);
    if (!stat) {
      llvm::errs() << "failed to parse static item in external item: "
                   << stat.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setStaticItem(stat.getValue());
    return StringResult<ast::ExternalItem>(impl);
  } else if (checkKeyWord(KeyWordKind::KW_CONST) ||
             checkKeyWord(KeyWordKind::KW_ASYNC) ||
             checkKeyWord(KeyWordKind::KW_EXTERN) ||
             checkKeyWord(KeyWordKind::KW_FN) ||
             checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    StringResult<std::shared_ptr<ast::Item>> fn =
        parseFunction(std::nullopt);
    if (!fn) {
      llvm::errs() << "failed to parse function item in external item: "
                   << fn.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setFunction(fn.getValue());
    return StringResult<ast::ExternalItem>(impl);
  } else if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
    // Macro
    StringResult<std::shared_ptr<ast::Expression>> macro =
        parseMacroInvocationExpression();
    if (!macro) {
      llvm::errs() << "failed to parse macro invocation expression item in "
                      "external item: "
                   << macro.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setMacroInvocation(macro.getValue());
    return StringResult<ast::ExternalItem>(impl);
  }
  return StringResult<ast::ExternalItem>("failed to parse external item");
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseExternBlock(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  ExternBlock impl = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
    impl.setUnsafe();
  }

  if (!checkKeyWord(KeyWordKind::KW_EXTERN))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse extern keyword in extern block");
  assert(eatKeyWord(KeyWordKind::KW_EXTERN));

  StringResult<ast::Abi> abi = parseAbi();
  if (!abi) {
    llvm::errs() << "failed to parse abi in "
                    "external block: "
                 << abi.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  impl.setAbi(abi.getValue());

  if (!check(TokenKind::BraceOpen))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse { token in extern block");
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (!innerAttributes) {
      llvm::errs() << "failed to parse inner attributes in "
                      "external block: "
                   << innerAttributes.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::InnerAttribute> inner = innerAttributes.getValue();
    impl.setInnerAttributes(inner);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (check(TokenKind::BraceClose)) {
      // done
      assert(eat(TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<ExternBlock>(impl));
    } else {
      StringResult<ast::ExternalItem> item = parseExternalItem();
      if (!item) {
        llvm::errs() << "failed to parse external item in "
                        "external block: "
                     << item.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      impl.addItem(item.getValue());
    }
  }

  return StringResult<std::shared_ptr<ast::Item>>(
      "failed to parse  extern block");
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseImplementation(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    recover(cp);
    return parseTraitImpl(vis);
  }

  if (!checkKeyWord(KeyWordKind::KW_IMPL))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse impl keyword in implementation");

  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> generic = parseGenericParams();
    if (!generic) {
      llvm::errs() << "failed to parse generic params in "
                      "implementation: "
                   << generic.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
  }

  if (check(TokenKind::Not)) {
    recover(cp);
    return parseTraitImpl(vis);
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> path = parseTypePath();
  if (path) {
    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      recover(cp);
      return parseTraitImpl(vis);
    }
  }

  recover(cp);
  return parseInherentImpl(vis);
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseTypeAlias(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  TypeAlias alias = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_TYPE))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse type keyword in type alias");

  assert(eatKeyWord(KeyWordKind::KW_TYPE));

  if (!check(TokenKind::Identifier))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier in type alias");

  Token id = getToken();
  alias.setIdentifier(id.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> genericParams = parseGenericParams();
    if (!genericParams) {
      llvm::errs() << "failed to parse generic params in "
                      "type alias "
                   << genericParams.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    alias.setGenericParams(genericParams.getValue());
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<TypeAlias>(alias));
  }

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    StringResult<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (!bounds) {
      llvm::errs() << "failed to parse type param bounds in "
                      "type alias "
                   << bounds.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    alias.setParamBounds(bounds.getValue());
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<TypeAlias>(alias));
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in "
                      "type alias "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    alias.setWhereClause(where.getValue());
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<TypeAlias>(alias));
  } else if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
  } else {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse where keyword or ; token in type alias");
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in "
                    "type alias "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  alias.setType(type.getValue());

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in "
                      "type alias "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    alias.setTypeWhereClause(where.getValue());
  } else if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<TypeAlias>(alias));
  }
  {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse where keyword or ; token in type alias");
  }

  assert(eat(TokenKind::Semi));
  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<TypeAlias>(alias));
}

  StringResult<std::shared_ptr<ast::Item>>
Parser::parseStaticItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  StaticItem stat = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STATIC))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse static keyword in static item");

  assert(eatKeyWord(KeyWordKind::KW_STATIC));

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    stat.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  }

  if (!check(TokenKind::Identifier))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier in static item");

  Token id = getToken();
  stat.setIdentifier(id.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Semi))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse : in static item");
  assert(eat(TokenKind::Semi));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (!typeExpr) {
    llvm::errs() << "failed to parse type in "
                    "static item "
                 << typeExpr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  stat.setType(typeExpr.getValue());

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<StaticItem>(stat));
  } else if (check(TokenKind::Eq)) {
    // initializer
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> init =
        parseExpression({}, restrictions);
    if (!init) {
      llvm::errs() << "failed to parse expression in "
                      "static item "
                   << init.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    stat.setInit(init.getValue());
    assert(eat(TokenKind::Semi));
  } else {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse static item");
  }
  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<StaticItem>(stat));
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseConstantItem(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  ConstantItem con = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_CONST))
    return StringResult<std::shared_ptr<ast::Item>>(
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
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier token in constant item");
  }

  if (!check(TokenKind::Colon)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse colon token in constant item");
  }

  assert(eat(TokenKind::Colon));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> typeExpr =
      parseType();
  if (!typeExpr) {
    llvm::errs() << "failed to parse type in "
                    "constant item "
                 << typeExpr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  con.setType(typeExpr.getValue());

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<ConstantItem>(con));
  } else if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    // initializer
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> init =
        parseExpression({}, restrictions);
    if (!init) {
      llvm::errs() << "failed to parse expression in "
                      "constant item "
                   << init.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    con.setInit(init.getValue());
    assert(eat(TokenKind::Semi));
  } else {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse constant item");
  }

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<ConstantItem>(con));
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseUnion(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Union uni = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_UNION))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse union keyword in union");

  assert(eatKeyWord(KeyWordKind::KW_UNION));

  if (!check(TokenKind::Identifier))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier token in union");

  assert(check(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> genericParams = parseGenericParams();
    if (!genericParams) {
      llvm::errs() << "failed to parse generic params in "
                      "union "
                   << genericParams.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    uni.setGenericParams(genericParams.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> whereClause = parseWhereClause();
    if (!whereClause) {
      llvm::errs() << "failed to parse where clause in "
                      "union "
                   << whereClause.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    uni.setWhereClause(whereClause.getValue());
  }

  if (!check(TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse { token in union");
  }
  assert(check(TokenKind::BraceOpen));

  StringResult<ast::StructFields> fields = parseStructFields();
  if (!fields) {
    llvm::errs() << "failed to parse struct fields in "
                    "union "
                 << fields.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  uni.setStructfields(fields.getValue());
  assert(check(TokenKind::BraceClose));

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<Union>(uni));
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseStruct(std::optional<ast::Visibility> vis) {
  CheckPoint cp = getCheckPoint();

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse struct keyword in struct");

  assert(checkKeyWord(KeyWordKind::KW_STRUCT));

  if (!check(TokenKind::Identifier)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier in struct");
  }
  assert(check(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> genericParams = parseGenericParams();
    if (!genericParams) {
      llvm::errs() << "failed to parse generic params in "
                      "struct: "
                   << genericParams.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> whereClause = parseWhereClause();
    if (!whereClause) {
      llvm::errs() << "failed to parse where clause in "
                      "struct: "
                   << whereClause.getError() << "\n";
      printFunctionStack();
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
