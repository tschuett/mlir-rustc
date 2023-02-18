#include "AST/Types/TupleType.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTupleType() {
  Location loc = getLocation();

  TupleType tuple = {loc};

  if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::ParenOpen));
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<TupleType>(tuple);
  } else if (!check(TokenKind::ParenOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in trait");
  }

  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> first =
      parseType();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse type in tuple type: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  tuple.addType(*first);

  if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
    assert(eat(TokenKind::ParenOpen));
    return std::make_shared<TupleType>(tuple);
  } else if (!check(TokenKind::ParenClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in trait");
  }

  assert(eat(TokenKind::Comma));

  while (true) {
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> typ =
        parseType();
    if (auto e = typ.takeError()) {
      llvm::errs() << "failed to parse type in tuple type: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tuple.addType(*typ);

    if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::ParenOpen));
      return std::make_shared<TupleType>(tuple);
    } else if (check(TokenKind::Eof)) {
      // abort
      ///
    } else if (check(TokenKind::ParenClose)) {
      return std::make_shared<TupleType>(tuple);
    } else if (check(TokenKind::Comma)) {
      continue;
    }
  }
}

} // namespace rust_compiler::parser
