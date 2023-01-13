#pragma once

#include "AST/Module.h"

#include <memory>

namespace rust_compiler::sema {

class Sema {
public:
  void analyze(std::shared_ptr<ast::Module> &ast);

private:
  void walkModule(std::shared_ptr<ast::Module> module);
  void walkItem(std::shared_ptr<ast::Item> item);
  void walkVisItem(std::shared_ptr<ast::VisItem> item);
  void walkOuterAttributes(std::shared_ptr<ast::OuterAttributes>);
};

void analyzeSemantics(std::shared_ptr<ast::Module> &module);

} // namespace rust_compiler::sema
