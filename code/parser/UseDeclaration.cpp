#include "AST/UseDeclaration.h"

#include "AST/UseTree.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Item>>
Parser::parseUseDeclaration(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();
  UseDeclaration use = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_USE)) {
    assert(eatKeyWord(KeyWordKind::KW_USE));
  } else {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse use keyword in use declarion");
  }

  StringResult<ast::use_tree::UseTree> tree =
      parseUseTree();
  if (!tree) {
    llvm::errs() << "failed to parse simple path in macro invocation item: "
                 << tree.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  use.setTree(tree.getValue());

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<UseDeclaration>(use));
}

} // namespace rust_compiler::parser
