#include "AST/Patterns/SlicePattern.h"

#include "AST/Patterns/SlicePatternItems.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::patterns::SlicePatternItems>
Parser::parseSlicePatternItems() {
  Location loc = getLocation();

  SlicePatternItems items = {loc};

  StringResult<std::shared_ptr<ast::patterns::Pattern>> first = parsePattern();
  if (!first) {
    llvm::errs() << "failed to parse first pattern in slice pattern items: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0}\n{1}",
                      "failed to parse first pattern in slice pattern items: ",
                      first.getError())
            .str();
    return StringResult<ast::patterns::SlicePatternItems>(s);
  }
  items.addPattern(first.getValue());
  if (check(TokenKind::Comma))
    assert(eat(TokenKind::Comma));

  while (true) {
    llvm::errs() << "parse slice pattern items: "
                 << Token2String(getToken().getKind()) << "\n";
    if (check(TokenKind::Eof)) {
      // report error
      return StringResult<ast::patterns::SlicePatternItems>(
          "failed to parse slice pattern items: eof");
    } else if (check(TokenKind::SquareClose)) {
      // done
      return StringResult<ast::patterns::SlicePatternItems>(items);
    } else if (check(TokenKind::Comma) and check(TokenKind::SquareClose, 1)) {
      assert(eat(TokenKind::Comma));
      // done
      return StringResult<ast::patterns::SlicePatternItems>(items);
    } else {
      StringResult<std::shared_ptr<ast::patterns::Pattern>> first =
          parsePattern();
      if (!first) {
        llvm::errs() << "failed to parse next pattern in slice pattern items: "
                     << first.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse next pattern in slice pattern items: ",
                first.getError())
                .str();
        return StringResult<ast::patterns::SlicePatternItems>(s);
      }
      items.addPattern(first.getValue());
      if (check(TokenKind::Comma))
        assert(eat(TokenKind::Comma));
    }
  }

  return StringResult<ast::patterns::SlicePatternItems>(items);
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseSlicePattern() {
  Location loc = getLocation();

  SlicePattern slice = {loc};

  if (!check(TokenKind::SquareOpen)) {
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse [ token in slice pattern");
  }
  assert(eat(TokenKind::SquareOpen));

  if (check(TokenKind::SquareClose)) {
    // done
    assert(eat(TokenKind::SquareClose));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<SlicePattern>(slice));
  }

  StringResult<ast::patterns::SlicePatternItems> items =
      parseSlicePatternItems();
  if (!items) {
    llvm::errs() << "failed to parse slice pattern items in slice pattern: "
                 << items.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  slice.setPatternItems(items.getValue());

  if (!check(TokenKind::SquareClose)) {
    // report error
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ] token in parse slice pattern");
  }
  assert(eat(TokenKind::SquareClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<SlicePattern>(slice));
}

} // namespace rust_compiler::parser
