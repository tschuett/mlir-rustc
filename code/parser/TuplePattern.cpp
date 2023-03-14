#include "AST/Patterns/TuplePattern.h"

#include "AST/Patterns/TuplePatternItems.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

StringResult<TuplePatternItems> Parser::parseTuplePatternItems() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  llvm::errs() << "parseTuplePatternItems"
               << "\n";

  Location loc = getLocation();
  TuplePatternItems items = {loc};

  if (check(TokenKind::DotDot)) {
    items.setRestPattern();
    return StringResult<TuplePatternItems>(items);
  }

  StringResult<std::shared_ptr<ast::patterns::Pattern>> first = parsePattern();
  if (!first) {
    llvm::errs() << "failed to parse first pattern in tuple pattern items: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse first pattern in tuple pattern items",
                      first.getError())
            .str();
    return StringResult<TuplePatternItems>(s);
    // exit(EXIT_FAILURE);
  }
  items.addPattern(first.getValue());

  if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    items.setTrailingComma();
    assert(eat(TokenKind::Comma));
    return StringResult<TuplePatternItems>(items);
  } else if (check(TokenKind::ParenClose)) {
    return StringResult<TuplePatternItems>(items);
  }

  assert(eat(TokenKind::Comma));

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<TuplePatternItems>(
          "failed to parse tuple pattern items: eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return StringResult<TuplePatternItems>(items);
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<TuplePatternItems>(items);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
    } else {
      StringResult<std::shared_ptr<ast::patterns::Pattern>> next =
          parsePattern();

      if (!next) {
        llvm::errs() << "failed to parse next pattern in tuple pattern items: "
                     << next.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv("{0} {1}",
                          "failed to parse next pattern in tuple pattern items",
                          next.getError())
                .str();
        return StringResult<TuplePatternItems>(s);
        // exit(EXIT_FAILURE);
      }
      items.addPattern(next.getValue());
    }
  }
  return StringResult<TuplePatternItems>("failed to parse tuple pattern items");
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTuplePattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TuplePattern tuple = {loc};

  llvm::errs() << "parseTuplePattern"
               << "\n";

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ( token in tuple pattern");
  assert(eat(TokenKind::ParenOpen));

  StringResult<TuplePatternItems> items = parseTuplePatternItems();
  if (!items) {
    llvm::errs() << "failed to parse tuple pattern items in tuple pattern: "
                 << items.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse tuple pattern items in tuple pattern: ",
                      items.getError())
            .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
    // exit(EXIT_FAILURE);
  }
  tuple.setItems(items.getValue());

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ) token in tuple pattern");
  assert(eat(TokenKind::ParenClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<TuplePattern>(tuple));
}

} // namespace rust_compiler::parser
