#include "AST/ConstantItem.h"
#include "AST/FunctionQualifiers.h"
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
    analyzeFunction(std::static_pointer_cast<Function>(item).get());
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
    analyzeTrait(std::static_pointer_cast<Trait>(item).get());
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

void Sema::analyzeTrait(ast::Trait *trait) {
  for (auto &asso : trait->getAssociatedItems()) {
    if (asso.hasFunction()) {
      std::shared_ptr<Function> itemFun = std::static_pointer_cast<Function>(
          std::static_pointer_cast<VisItem>(asso.getFunction()));
      FunctionQualifiers qual = itemFun->getQualifiers();
      if (qual.hasAsync())
        llvm::errs() << "error: " << itemFun->getName().toString() << "trait"
                     << trait->getIdentifier().toString() << " is async"
                     << "\n";
      if (qual.hasConst())
        llvm::errs() << "error: " << itemFun->getName().toString() << "trait"
                     << trait->getIdentifier().toString() << " is const"
                     << "\n";
    }
  }
}

} // namespace rust_compiler::sema
