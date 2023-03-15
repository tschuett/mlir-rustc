#include "ItemVisitor.h"

#include "AST/BlockExpression.h"
#include "AST/Enumeration.h"
#include "AST/ExternBlock.h"
#include "AST/ExternCrate.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/Statements.h"
#include "AST/Struct.h"
#include "AST/Trait.h"
#include "AST/TraitImpl.h"
#include "AST/TypeAlias.h"
#include "AST/UseDeclaration.h"
#include "AST/Union.h"
#include "AST/ConstantItem.h"
#include "AST/StaticItem.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void visitItem(std::shared_ptr<ast::Item> item, ItemVisitor *visitor);

void visitVisItem(std::shared_ptr<ast::VisItem> visItem, ItemVisitor *visitor);

void visitStatements(ast::Statements stmts, ItemVisitor *visitor);

void visitAssociatedItem(ast::AssociatedItem assoItem, ItemVisitor *visitor);

void visitFunction(std::shared_ptr<ast::Function> fun, ItemVisitor *visitor);
void visitStruct(std::shared_ptr<ast::Struct> fun, ItemVisitor *visitor);
void visitEnumeration(std::shared_ptr<ast::Enumeration> fun,
                      ItemVisitor *visitor);
void visitTypeAlias(std::shared_ptr<ast::TypeAlias> fun, ItemVisitor *visitor);
void visitUseDeclaration(std::shared_ptr<ast::UseDeclaration> fun, ItemVisitor *visitor);
void visitExternCrate(std::shared_ptr<ast::ExternCrate> fun, ItemVisitor *visitor);
void visitUnion(std::shared_ptr<ast::Union> fun, ItemVisitor *visitor);
void visitConstantItem(std::shared_ptr<ast::ConstantItem> fun, ItemVisitor *visitor);
void visitStaticItem(std::shared_ptr<ast::StaticItem> fun, ItemVisitor *visitor);

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
  case StatementKind::EmptyStatement: {
    break;
  }
  case StatementKind::ItemDeclaration: {
    break;
  }
  case StatementKind::LetStatement: {
    break;
  }
  case StatementKind::ExpressionStatement: {
    break;
  }
  case StatementKind::MacroInvocationSemi: {
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
}

void visitTrait(std::shared_ptr<ast::Trait> trait, ItemVisitor *visitor) {
  for (auto &assoItem : trait->getAssociatedItems()) {
    visitAssociatedItem(assoItem, visitor);
  }
}

void visitBlockExpression(std::shared_ptr<ast::Expression> block,
                          ItemVisitor *visitor) {
  std::shared_ptr<ast::BlockExpression> _block =
      std::static_pointer_cast<BlockExpression>(block);
  visitStatements(_block->getExpressions(), visitor);
}

void visitFunction(std::shared_ptr<ast::Function> fun, ItemVisitor *visitor) {
  if (fun->hasBody())
    visitBlockExpression(fun->getBody(), visitor);
}

void visitStatements(ast::Statements stmts, ItemVisitor *visitor) {
  for (auto &stmt : stmts.getStmts())
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
  case VisItemKind::TypeAlias: {
    visitTypeAlias(std::static_pointer_cast<TypeAlias>(visItem), visitor);
    break;
  }
  case VisItemKind::UseDeclaration: {
    visitUseDeclaration(std::static_pointer_cast<UseDeclaration>(visItem),
                        visitor);
    break;
  }
  case VisItemKind::ExternCrate: {
    visitExternCrate(std::static_pointer_cast<ExternCrate>(visItem), visitor);
    break;
  }
  case VisItemKind::Struct: {
    visitStruct(std::static_pointer_cast<Struct>(visItem), visitor);
    break;
  }
  case VisItemKind::Enumeration: {
    visitEnumeration(std::static_pointer_cast<Enumeration>(visItem), visitor);
    break;
  }
  case VisItemKind::Union: {
    visitUnion(std::static_pointer_cast<Union>(visItem), visitor);
    break;
  }
  case VisItemKind::ConstantItem: {
    visitConstantItem(std::static_pointer_cast<ConstantItem>(visItem), visitor);
    break;
  }
  case VisItemKind::StaticItem: {
    visitStaticItem(std::static_pointer_cast<StaticItem>(visItem), visitor);
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
