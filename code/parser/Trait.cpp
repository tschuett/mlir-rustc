#include "AST/Trait.h"

#include "AST/TraitImpl.h"
#include "AST/InherentImpl.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseInherentImpl(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  InherentImpl impl = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse impl keyword in inherent impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> generic = parseGenericParams();
    if (auto e = generic.takeError()) {
      llvm::errs() << "failed to parse generic params in trait impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setGenericParams(*generic);
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in inherent impl: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  impl.setType(*type);

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in inherent impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setWhereClause(*where);
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in inherent impl");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (auto e = inner.takeError()) {
      llvm::errs() << "failed to parse inner attributes in inherent impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setInnerAttributes(*inner);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // error
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse inherent impl: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return std::make_shared<InherentImpl>(impl);
    } else if (!check(TokenKind::BraceClose)) {
      // asso without check
      llvm::Expected<ast::AssociatedItem> asso = parseAssociatedItem();
      if (auto e = asso.takeError()) {
        llvm::errs() << "failed to parse associated item in inherent impl: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      impl.addAssociatedItem(*asso);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse inherent impl");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse inherent impl");
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseTraitImpl(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  TraitImpl impl = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    impl.setUnsafe();
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  }

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse impl keyword in trait impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> generic = parseGenericParams();
    if (auto e = generic.takeError()) {
      llvm::errs() << "failed to parse generic params in trait impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setGenericParams(*generic);
  }

  if (check(TokenKind::Not)) {
    assert(eat(TokenKind::Not));
    impl.setNot();
  }

  llvm::Expected<std::shared_ptr<ast::types::TypePath>> typePath =
      parseTypePath();
  if (auto e = typePath.takeError()) {
    llvm::errs() << "failed to parse type path in trait impl: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  impl.setTypePath(*typePath);

  if (!checkKeyWord(KeyWordKind::KW_FOR)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse for keyword in trait impl");
  }
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in trait impl: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  impl.setType(*type);

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in trait impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setWhereClause(*where);
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in trait impl");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (auto e = inner.takeError()) {
      llvm::errs() << "failed to parse inner attributes in trait impl: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    impl.setInnerAttributes(*inner);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      // error
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse trait impl: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      return std::make_shared<TraitImpl>(impl);
    } else if (!check(TokenKind::BraceClose)) {
      // asso without check
      llvm::Expected<ast::AssociatedItem> asso = parseAssociatedItem();
      if (auto e = asso.takeError()) {
        llvm::errs() << "failed to parse associated item in trait impl: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      impl.addAssociatedItem(*asso);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse trait impl");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse trait impl");
}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseTrait(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Trait trait = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    trait.setUnsafe();
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
  }

  if (!checkKeyWord(KeyWordKind::KW_TRAIT)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse trait keyword in trait");
  }
  assert(eatKeyWord(KeyWordKind::KW_TRAIT));

  if (!check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in trait");
  }
  trait.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> params = parseGenericParams();
    if (auto e = params.takeError()) {
      llvm::errs() << "failed to parse generic params in trait: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    trait.setGenericParams(*params);
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
    llvm::Expected<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
    if (auto e = bounds.takeError()) {
      llvm::errs() << "failed to parse generic params in trait: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    assert(eat(TokenKind::ParenClose));
    trait.setBounds(*bounds);
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> where = parseWhereClause();
    if (auto e = where.takeError()) {
      llvm::errs() << "failed to parse where clause in trait: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    trait.setWhere(*where);
  }

  if (!check(TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse { token in trait");
  }
  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    llvm::Expected<std::vector<ast::InnerAttribute>> inner =
        parseInnerAttributes();
    if (auto e = inner.takeError()) {
      llvm::errs() << "failed to parse inner attributes in trait: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    trait.setInner(*inner);
  }

  // FIXME
  // xxx;

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse trait: eof");
    } else if (check(TokenKind::BraceClose)) {
      // done
      assert(eat(TokenKind::BraceClose));
      return std::make_shared<Trait>(trait);
    } else {
      llvm::Expected<ast::AssociatedItem> asso = parseAssociatedItem();
      if (auto e = asso.takeError()) {
        llvm::errs() << "failed to parse associated item in trait: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      trait.addItem(*asso);
    }
  }

  if (!check(TokenKind::BraceClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse } token in trait");
  }
  assert(eat(TokenKind::BraceClose));

  return std::make_shared<Trait>(trait);
}

} // namespace rust_compiler::parser
