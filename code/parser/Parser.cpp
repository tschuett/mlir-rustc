#include "Parser/Parser.h"

#include "AST/ClippyAttribute.h"
#include "AST/Crate.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "Lexer/Token.h"
#include "Util.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace mlir;

namespace rust_compiler::parser {

LogicalResult Parser::parseFile(std::shared_ptr<ast::Module> &module) {
  std::span<Token> tokens = ts.getAsView();

  size_t last = tokens.size();
  while (tokens.size() > 0) {
    last = tokens.size();

    // printTokenState(tokens);

    std::optional<std::shared_ptr<ast::Item>> item = tryParseItem(tokens);
    if (item) {
      llvm::errs() << "found tokens: " << (*item)->getTokens() << "\n";
      tokens = tokens.subspan((*item)->getTokens());
      module->addItem(*item);
      llvm::errs() << "added item"
                   << "\n";
    } else {
      return LogicalResult::failure();
    }

    if (tokens.size() == last) {
      llvm::errs() << "parser: no progress"
                   << "\n";
      printTokenState(tokens);
      exit(EXIT_FAILURE);
    }
  }

  return LogicalResult::success();
}

} // namespace rust_compiler::parser
