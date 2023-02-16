#include "AST/Trait.h"

#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

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
  Token tok = getToken();
  trait.setIdentifier(tok.getIdentifier());
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
}

} // namespace rust_compiler::parser
