#include "AST/Struct.h"
#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace llvm;
using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseStruct(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  // StructStruct stru = {loc, vis};

  if (!checkKeyWord(KeyWordKind::KW_STRUCT))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct keyword in struct");

  if (check(TokenKind::Identifier)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier in struct");
  }

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
    // check error
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClasue = parseWhereClause();
    // check error
  }

  if (check(TokenKind::BraceOpen)) {
    // StructStruct
  }

  if (check(TokenKind::ParenOpen)) {
    // TupleStruct
  }

  // FIXME
}

} // namespace rust_compiler::parser
