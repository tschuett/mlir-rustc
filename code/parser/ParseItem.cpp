#include "AST/OuterAttribute.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace llvm;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Item>> Parser::parseItem() {
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attribute in item : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }

    if (checkVisItem()) {
      return parseVisItem(*outer);
    } else if (checkMacroItem()) {
      return parseMacroItem(*outer);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse item");
    }
  }

  std::vector<OuterAttribute> outer;
  if (checkVisItem()) {
    return parseVisItem(outer);
  } else if (checkMacroItem()) {
    return parseMacroItem(outer);
  }
  return createStringError(inconvertibleErrorCode(), "failed to parse item");
}

} // namespace rust_compiler::parser
