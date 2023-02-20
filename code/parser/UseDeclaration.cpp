#include "AST/UseDeclaration.h"

#include "AST/UseTree.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseUseDeclaration(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  UseDeclaration use = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_USE)) {
    assert(eatKeyWord(KeyWordKind::KW_USE));
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse use keyword in use declarion");
  }

  llvm::Expected<ast::use_tree::UseTree> tree = parseUseTree();
  if (auto e = tree.takeError()) {
    llvm::errs() << "failed to use tree in use declaration: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  use.setTree(*tree);

  return std::make_shared<UseDeclaration>(use);
}


} // namespace rust_compiler::parser
