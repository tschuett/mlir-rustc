#pragma once

#include "AST/AST.h"
#include "AST/TokenTree.h"
#include "Lexer/TokenStream.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

enum class DelimTokenTreeKind { Paren, Square, Brace };

class DelimTokenTree : public Node {
  DelimTokenTreeKind kind;
  std::vector<TokenTree> trees;

public:
  DelimTokenTree(Location loc) : Node(loc){};

  void setKind(DelimTokenTreeKind kind_) { kind = kind_; }
  void addTree(const TokenTree &t) { trees.push_back(t); }

  bool isEmpty() const;
  DelimTokenTreeKind getKind() const { return kind; }

  size_t getNumberOfTrees() const { return trees.size(); }

  std::vector<lexer::Token> toTokenStream();
};

} // namespace rust_compiler::ast
