#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/TypeNoBounds.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseRawPointerType() {
  Location loc = getLocation();
  RawPointerType rawPointer = {loc};

  if (!check(TokenKind::Star))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse * token in raw pointer type");
  assert(eat(TokenKind::Star));

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    rawPointer.setMut();
  } else if (checkKeyWord(KeyWordKind::KW_CONST)) {
    rawPointer.setConst();
  } else {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse mut or const keyword in raw pointer type");
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (auto e = noBounds.takeError()) {
    llvm::errs() << "failed to parse type in raw pointer type: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  rawPointer.setType(*noBounds);

  return std::make_shared<RawPointerType>(rawPointer);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseReferenceType() {
  Location loc = getLocation();
  ReferenceType refType = {loc};

  if (!check(TokenKind::And))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse & token in reference type");
  assert(eat(TokenKind::And));

  // FIXME Lifetime

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    refType.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (auto e = noBounds.takeError()) {
    llvm::errs() << "failed to parse type in reference type: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  refType.setType(*noBounds);

  return std::make_shared<ReferenceType>(refType);
}

} // namespace rust_compiler::parser
