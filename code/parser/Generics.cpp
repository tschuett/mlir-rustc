#include "Generics.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::lexer {

std::optional<std::shared_ptr<GenericParams>>
tryParseGenericParams(std::span<Token> tokens) {
  // FIXME
  return std::nullopt;
  
}

}
