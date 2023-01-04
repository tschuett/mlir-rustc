#include "Generics.h"

namespace rust_compiler::lexer {

std::optional<std::shared_ptr<GenericParams>>
tryParseGenericParams(std::span<lexer::Token> tokens);

}
