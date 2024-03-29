#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/TypeNoBounds.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseRawPointerType() {
  Location loc = getLocation();
  RawPointerType rawPointer = {loc};

  if (!check(TokenKind::Star))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse * token in raw pointer type");
  assert(eat(TokenKind::Star));

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    rawPointer.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  } else if (checkKeyWord(KeyWordKind::KW_CONST)) {
    assert(eatKeyWord(KeyWordKind::KW_CONST));
    rawPointer.setConst();
  } else {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(

        "failed to parse mut or const keyword in raw pointer type");
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (!noBounds) {
    llvm::errs() << "failed to parse type no bounds in raw pointer type: "
                 << noBounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  rawPointer.setType(noBounds.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<RawPointerType>(rawPointer));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseReferenceType() {
  Location loc = getLocation();
  ReferenceType refType = {loc};

  //  llvm::errs() << "parseReferenceType"
  //               << "\n";

  if (!check(TokenKind::And))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse & token in reference type");
  assert(eat(TokenKind::And));

  // FIXME Lifetime

  if (check(TokenKind::LIFETIME_OR_LABEL) or
      checkKeyWord(KeyWordKind::KW_STATICLIFETIME) or
      (check(TokenKind::LIFETIME_TOKEN) && getToken().getStorage() == "'_")) {
    adt::Result<ast::Lifetime, std::string> lifetime =
        parseLifetimeAsLifetime();
    if (!lifetime) {
      llvm::errs() << "failed to parse lifetime in reference tpye: "
                   << lifetime.getError() << "\n";
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse lifetime in reference type: ",
                        lifetime.getError())
              .str();
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
    }
    refType.setLifetime(lifetime.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    refType.setMut();
    assert(eatKeyWord(KeyWordKind::KW_MUT));
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> noBounds =
      parseTypeNoBounds();
  if (!noBounds) {
    llvm::errs() << "failed to parse type no bounds in reference type: "
                 << noBounds.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse type no bounds in reference type: ",
                      noBounds.getError())
            .str();
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
  }

  refType.setType(noBounds.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<ReferenceType>(refType));
}

} // namespace rust_compiler::parser
