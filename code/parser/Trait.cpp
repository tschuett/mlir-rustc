#include "AST/Trait.h"

#include "ADT/Result.h"
#include "AST/InherentImpl.h"
#include "AST/TraitImpl.h"
#include "AST/Types/TypeExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormatVariadic.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Item>>
Parser::parseInherentImpl(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  InherentImpl impl = {loc, vis};

  //llvm::errs() << "parseInherentImpl"
  //             << "\n";

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse impl keyword in inherent impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> generic = parseGenericParams();
    if (!generic) {
      llvm::errs() << "failed to parse generic params in inherent impl: "
                   << generic.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setGenericParams(generic.getValue());
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in inherent impl: " << type.getError()
                 << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv("{0} {1}", "failed to parse type in inherent impl: ",
                      type.getError())
            .str();
    return StringResult<std::shared_ptr<ast::Item>>(s);
  }
  impl.setType(type.getValue());

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in inherent impl: "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setWhereClause(where.getValue());
  }

  if (!check(TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse { token in inherent impl");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (!inner) {
      llvm::errs() << "failed to parse inner attributes in inherent impl: "
                   << inner.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::InnerAttribute> in = inner.getValue();
    impl.setInnerAttributes(in);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // error
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse inherent impl: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      assert(eat(TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<InherentImpl>(impl));
    } else if (!check(TokenKind::BraceClose)) {
      // asso without check
      StringResult<ast::AssociatedItem> asso = parseAssociatedItem();
      if (!asso) {
        llvm::errs() << "failed to parse associated item in inherent impl: "
                     << asso.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      impl.addAssociatedItem(asso.getValue());
    } else {
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse inherent impl");
    }
  }
  return StringResult<std::shared_ptr<ast::Item>>(
      "failed to parse inherent impl");
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseTraitImpl(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  TraitImpl impl = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    impl.setUnsafe();
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  }

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse impl keyword in trait impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> generic = parseGenericParams();
    if (!generic) {
      llvm::errs() << "failed to parse generic params item in trait impl: "
                   << generic.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setGenericParams(generic.getValue());
  }

  if (check(TokenKind::Not)) {
    assert(eat(TokenKind::Not));
    impl.setNot();
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> typePath =
      parseTypePath();
  if (!typePath) {
    llvm::errs() << "failed to parse type item in trait impl: "
                 << typePath.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  impl.setTypePath(typePath.getValue());

  if (!checkKeyWord(KeyWordKind::KW_FOR)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse for keyword in trait impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in trait impl: " << type.getError()
                 << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  impl.setType(type.getValue());

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in trait impl: "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    impl.setWhereClause(where.getValue());
  }

  if (!check(TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse { token in trait impl");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (!inner) {
      llvm::errs() << "failed to parse inner attribute in trait impl: "
                   << inner.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::InnerAttribute> in = inner.getValue();
    impl.setInnerAttributes(in);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // error
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse trait impl: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<TraitImpl>(impl));
    } else if (!check(TokenKind::BraceClose)) {
      // asso without check
      StringResult<ast::AssociatedItem> asso = parseAssociatedItem();
      if (!asso) {
        llvm::errs() << "failed to parse associated item in trait impl: "
                     << asso.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      impl.addAssociatedItem(asso.getValue());
    } else {
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse trait impl");
    }
  }
  return StringResult<std::shared_ptr<ast::Item>>("failed to parse trait impl");
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseTrait(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Trait trait = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    trait.setUnsafe();
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  }

  if (!checkKeyWord(KeyWordKind::KW_TRAIT)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse trait keyword in trait");
  }
  assert(eatKeyWord(KeyWordKind::KW_TRAIT));

  if (!check(TokenKind::Identifier)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier token in trait");
  }
  trait.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> params = parseGenericParams();
    if (!params) {
      llvm::errs() << "failed to parse generic params in trait impl: "
                   << params.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    trait.setGenericParams(params.getValue());
  }

  if (check(TokenKind::ParenOpen) && check(TokenKind::Colon, 1) &&
      check(TokenKind::ParenClose, 2)) {
    assert(eat(TokenKind::ParenOpen));
    assert(eat(TokenKind::Colon));
    assert(eat(TokenKind::ParenClose));
  } else if (check(TokenKind::ParenOpen) && check(TokenKind::Colon, 1) &&
             !check(TokenKind::ParenClose, 2)) {
    assert(eat(TokenKind::ParenOpen));
    assert(eat(TokenKind::Colon));
    StringResult<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (!bounds) {
      llvm::errs() << "failed to parse type param bounds in trait impl: "
                   << bounds.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    assert(eat(TokenKind::ParenClose));
    trait.setBounds(bounds.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> where = parseWhereClause();
    if (!where) {
      llvm::errs() << "failed to parse where clause in trait impl: "
                   << where.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    trait.setWhere(where.getValue());
  }

  if (!check(TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse { token in trait");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (!inner) {
      llvm::errs() << "failed to parse inner attribute in trait impl: "
                   << inner.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::InnerAttribute> in = inner.getValue();
    trait.setInner(in);
  }

  // FIXME
  // xxx;

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<std::shared_ptr<ast::Item>>(
          "failed to parse trait: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      assert(eat(TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::Item>>(
          std::make_shared<Trait>(trait));
    } else {
      StringResult<ast::AssociatedItem> asso = parseAssociatedItem();
      if (!asso) {
        llvm::errs() << "failed to parse associated item in trait impl: "
                     << asso.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      trait.addItem(asso.getValue());
    }
  }

  if (!check(TokenKind::BraceClose)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse } token in trait");
  }
  assert(eat(TokenKind::BraceClose));

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<Trait>(trait));
}

} // namespace rust_compiler::parser
