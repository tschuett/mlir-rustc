#include "Toml/TokenStream.h"

#include <algorithm>

namespace rust_compiler::toml {

void TokenStream::append(Token tok) { tokens.push_back(tok); }

} // namespace rust_compiler::toml
