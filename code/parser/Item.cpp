#include "AST/Function.h"
#include "AST/Item.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Item>>
Parser::tryParseItem(std::span<Token> tokens) {

  std::span<Token> view = tokens;

  //  llvm::errs() << "tryParseItem"
  //               << "\n";

  std::optional<std::shared_ptr<ast::OuterAttributes>> outer =
      tryParseOuterAttributes(view);

  if (outer) {
    view = view.subspan((*outer)->getTokens());

    std::optional<std::shared_ptr<ast::VisItem>> visItem =
        tryParseVisItem(view);

    if (visItem) {

      Item item{tokens.front().getLocation(), ItemKind::VisItem};
      item.setOuterAttributes(*outer);
      item.setVisItem(*visItem);
      return std::make_shared<ast::Item>(item);
    }
  } else {
    std::optional<std::shared_ptr<ast::VisItem>> visItem =
        tryParseVisItem(view);

    if (visItem) {

      Item item{tokens.front().getLocation(), ItemKind::VisItem};
      item.setVisItem(*visItem);
      return std::make_shared<ast::Item>(item);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser


// FIXME MacroItem
