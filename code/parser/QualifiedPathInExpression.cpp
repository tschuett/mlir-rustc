#include "QualifiedPathInExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseQualifiedPathInExpression(std::span<lexer::Token> tokens);

}
