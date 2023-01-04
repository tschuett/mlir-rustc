#include "Type.h

namespace rust_compiler::lexer {

  std::optional<std::shared_ptr<Type>> tryParseType(std::span<Token> tokens);

}
