#pragma once

#include "AST/Crate.h"
#include "AST/Item.h"
#include "AST/ItemDeclaration.h"

namespace rust_compiler::sema {

class ItemVisitor {
public:
  virtual ~ItemVisitor() = default;

  virtual void visitItem(std::shared_ptr<ast::Item> let) {}
  virtual void visitItemDeclaration(std::shared_ptr<ast::ItemDeclaration> let);
  virtual void visitVisItem(std::shared_ptr<ast::VisItem> let);
};

void run(std::shared_ptr<ast::Crate> crate, ItemVisitor *visitor);

} // namespace rust_compiler::sema
