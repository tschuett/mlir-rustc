#include "AST/ClippyAttribute.h"
#include "AST/Function.h"
#include "AST/UseDeclaration.h"
#include "AST/Visiblity.h"
#include "Attributes.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Item>>
Parser::tryParseItem(std::span<Token> tokens, std::string_view modulePath) {

  std::span<Token> view = tokens;

  //  llvm::errs() << "tryParseItem"
  //               << "\n";

  std::optional<std::shared_ptr<ast::OuterAttributes>> outer =
      tryParseOuterAttributes(view, modulePath);

  if (outer) {
    view = view.subspan((*outer)->getTokens());

    std::optional<std::shared_ptr<ast::VisItem>> visItem =
        tryParseVisItem(view, modulePath);

    if (visItem) {

      Item item{tokens.front().getLocation()};
      item.setOuterAttributes(*outer);
      item.setVisItem(*visItem);
      return std::make_shared<ast::Item>(item);
    }
  } else {
    std::optional<std::shared_ptr<ast::VisItem>> visItem =
        tryParseVisItem(view, modulePath);

    if (visItem) {

      Item item{tokens.front().getLocation()};
      item.setVisItem(*visItem);
      return std::make_shared<ast::Item>(item);
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
