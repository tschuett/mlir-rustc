#include "AST/DelimTokenTree.h"

#include "AST/TokenTree.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<ast::TokenTree> Parser::parseTokenTree() {
  Location loc = getLocation();
  TokenTree tree = {loc};
}

llvm::Expected<ast::DelimTokenTree> Parser::parseDelimTokenTree() {
  Location loc = getLocation();
  DelimTokenTree tree = {loc};
}

} // namespace rust_compiler::parser
