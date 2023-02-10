#include "ItemVisitor.h"

#include "AST/BlockExpression.h"
#include "AST/ExternBlock.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/Statements.h"
#include "AST/Trait.h"
#include "AST/TraitImpl.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void visitItem(std::shared_ptr<ast::Item> item, ItemVisitor *visitor);

void visitVisItem(std::shared_ptr<ast::VisItem> visItem, ItemVisitor *visitor);

void visitStatements(std::shared_ptr<ast::Statements> stmts,
                     ItemVisitor *visitor);

void visitAssociatedItem(ast::AssociatedItem assoItem, ItemVisitor *visitor);

void visitFunction(std::shared_ptr<ast::Function> fun, ItemVisitor *visitor);

///

void visitInherentImpl(std::shared_ptr<InherentImpl> impl,
                       ItemVisitor *visitor) {
  for (auto &asso : impl->getAssociatedItems())
    visitAssociatedItem(asso, visitor);
}

void visitTraitImpl(std::shared_ptr<TraitImpl> impl, ItemVisitor *visitor) {
  for (auto &asso : impl->getAssociatedItems())
    visitAssociatedItem(asso, visitor);
}

void visitStatement(std::shared_ptr<ast::Statement> stmt,
                    ItemVisitor *visitor) {

  // FIXME
  switch (stmt->getKind()) {
  case StatementKind::ItemDeclaration: {
    break;
  }
  case StatementKind::ExpressionStatement: {
    break;
  }
  }
}

void visitExternBlock(std::shared_ptr<ast::ExternBlock> externBlock,
                      ItemVisitor *visitor) {
  // FIXEM
}
void visitImplementation(std::shared_ptr<ast::Implementation> impl,
                         ItemVisitor *visitor) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    visitInherentImpl(std::static_pointer_cast<InherentImpl>(impl), visitor);
    break;
  }
  case ImplementationKind::TraitImpl: {
    visitTraitImpl(std::static_pointer_cast<TraitImpl>(impl), visitor);
    break;
  }
  }
}

void visitAssociatedItem(std::shared_ptr<ast::AssociatedItem> assoItem,
                         ItemVisitor *visitor) {
  switch (assoItem->getKind()) {
  case AssociatedItemKind::MacroInvocation: {
    break;
  }
  case AssociatedItemKind::TypeAlias: {
    break;
  }
  case AssociatedItemKind::ConstantItem: {
    break;
  }
  case AssociatedItemKind::Function: {
    visitFunction(std::static_pointer_cast<AssociatedItemFunction>(assoItem)
                      ->getFunction(),
                  visitor);
    break;
  }
  }
}

void visitTrait(std::shared_ptr<ast::Trait> trait, ItemVisitor *visitor) {
  for (auto &assoItem : trait->getAssociatedItems()) {
    visitAssociatedItem(assoItem, visitor);
  }
}

void visitBlockExpression(std::shared_ptr<ast::BlockExpression> block,
                          ItemVisitor *visitor) {
  visitStatements(block->getExpressions(), visitor);
}

void visitFunction(std::shared_ptr<ast::Function> fun, ItemVisitor *visitor) {
  if (fun->hasBody())
    visitBlockExpression(fun->getBody(), visitor);
}

void visitStatements(std::shared_ptr<ast::Statements> stmts,
                     ItemVisitor *visitor) {
  for (auto &stmt : stmts->getStmts())
    visitStatement(stmt, visitor);
}

void visitModule(std::shared_ptr<ast::Module> module, ItemVisitor *visitor) {
  for (auto &it : module->getItems())
    visitItem(it, visitor);
}

void visitVisItem(std::shared_ptr<ast::VisItem> visItem, ItemVisitor *visitor) {
  visitor->visitItem(visItem);

  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    visitModule(std::static_pointer_cast<Module>(visItem), visitor);
    break;
  }
  case VisItemKind::Function: {
    visitFunction(std::static_pointer_cast<Function>(visItem), visitor);
    break;
  }
  case VisItemKind::Implementation: {
    visitImplementation(std::static_pointer_cast<Implementation>(visItem),
                        visitor);
    break;
  }
  case VisItemKind::ExternBlock: {
    visitExternBlock(std::static_pointer_cast<ExternBlock>(visItem), visitor);
    break;
  }
  case VisItemKind::Trait: {
    visitTrait(std::static_pointer_cast<Trait>(visItem), visitor);
    break;
  }
  }
}

void visitItem(std::shared_ptr<ast::Item> item, ItemVisitor *visitor) {
  visitor->visitItem(item);
  visitVisItem(item->getVisItem(), visitor);
}

void run(std::shared_ptr<ast::Crate> crate, ItemVisitor *visitor) {
  for (auto &it : crate->getItems()) {
    visitItem(it, visitor);
  }
}

} // namespace rust_compiler::sema
