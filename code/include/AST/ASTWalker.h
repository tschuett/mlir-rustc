#pragma once

namespace rust_compiler::ast {
class Function;
class Item;
class Module;

class ASTWalker {
public:
  ASTWalker(Module *m);

  virtual ~ASTWalker() = default;

  virtual void visitItem(Item *) = 0;
  virtual void visitFunction(Function *) = 0;
};

} // namespace rust_compiler::ast
