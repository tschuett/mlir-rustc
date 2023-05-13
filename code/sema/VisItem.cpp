#include "AST/ConstantItem.h"
#include "AST/Module.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::walkVisItem(std::shared_ptr<ast::VisItem> item) {
  switch (item->getKind()) {
  case VisItemKind::Module: {
    break;
  }
  case VisItemKind::Function: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::TypeAlias: {
    break;
  }
  case VisItemKind::Struct: {
    break;
  }
  case VisItemKind::ConstantItem: {
    analyzeConstantItem(std::static_pointer_cast<ConstantItem>(item).get());
    break;
  }
  case VisItemKind::Enumeration: {
    break;
  }
  case VisItemKind::Union: {
    break;
  }
  case VisItemKind::StaticItem: {
    analyzeStaticItem(std::static_pointer_cast<StaticItem>(item).get());
    break;
  }
  case VisItemKind::Trait: {
    break;
  }
  case VisItemKind::Implementation: {
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

void Sema::analyzeConstantItem(ast::ConstantItem *ci) {
  if (ci->hasInit()) {
    [[maybe_unused]] bool isConst = isConstantExpression(ci->getInit().get());
  }

  walkType(ci->getType().get());
}

void Sema::analyzeStaticItem(StaticItem *si) {
  if (si->hasInit()) {
    [[maybe_unused]] bool isConst = isConstantExpression(si->getInit().get());
  }
  walkType(si->getType().get());
}

} // namespace rust_compiler::sema
