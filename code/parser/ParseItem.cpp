#include "AST/OuterAttribute.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Item>> Parser::parseItem() {
  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in item: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }

    std::vector<OuterAttribute> ot = outer.getValue();
    if (checkVisItem()) {
      return parseVisItem(ot);
    } else if (checkMacroItem()) {
      return parseMacroItem(ot);
    } else {
      return StringResult<std::shared_ptr<ast::Item>>("failed to parse item2");
    }
  }

  std::vector<OuterAttribute> outer;
  if (checkVisItem()) {
    return StringResult<std::shared_ptr<ast::Item>>(parseVisItem(outer));
  } else if (checkMacroItem()) {
    return parseMacroItem(outer);
  }

  return StringResult<std::shared_ptr<ast::Item>>("failed to parse item1");
}

} // namespace rust_compiler::parser
