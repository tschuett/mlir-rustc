#include "AST/Types/BareFunctionType.h"

#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseBareFunctionType() {
  Location loc = getLocation();

  BareFunctionType bare = {loc};

  // FIXME
}

} // namespace rust_compiler::parser
