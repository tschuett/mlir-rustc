#include "AST/DelimTokenTree.h"

namespace rust_compiler::ast {

bool DelimTokenTree::isEmpty() const { return trees.size() == 0; }

lexer::TokenStream DelimTokenTree::toTokenStream() { assert(false); }

} // namespace rust_compiler::ast
