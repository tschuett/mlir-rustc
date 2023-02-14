#include "Lexer/KeyWords.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseClosureExpression() {
  Location loc = getLocation();

  bool move = false;

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    move = true;
    assert(eatKeyWord(KeyWordKind::KW_MOVE));
  }

  
}

} // namespace rust_compiler::parser
