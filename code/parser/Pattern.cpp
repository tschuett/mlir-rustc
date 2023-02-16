#include "AST/Patterns/SlicePattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

llvm::Expected<ast::patterns::SlicePatternItems>
Parser::parseSlicePatternItems() {}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseSlicePattern() {
  Location loc = getLocation();

  SlicePattern slice = {loc};

  if (!check(TokenKind::SquareOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse [ token in slice pattern");
  }
  assert(check(TokenKind::SquareOpen));

  if (check(TokenKind::SquareClose)) {
    // done
  }

  llvm::Expected<ast::patterns::SlicePatternItems> items =
    parseSlicePatternItems();

  // check error

  // return
}

} // namespace rust_compiler::parser
