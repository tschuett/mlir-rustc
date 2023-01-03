#include "Item.h"

#include "AST/Visiblity.h"

namespace rust_compiler::parser {

std::optional<ast::Item> tryParseItem(std::span<Token> tokens,
                                      std::string_view modulePath) {

  std::span<Token> view = tokens;

  std::optional<ast::Visibility> visibility = tryParseVisibility(tokens);

}

} // namespace rust_compiler::parser
