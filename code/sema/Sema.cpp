#include "Sema/Sema.h"

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/Module.h"
#include "AST/VisItem.h"
#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Crate> &ast) {

  Sema sema;

  sema.analyze(ast);
}

void Sema::analyze(std::shared_ptr<ast::Crate> &ast) {
  // FIXME

  for (const auto &item : ast->getItems()) {
    const auto &visItem = item->getVisItem();
    switch (visItem->getKind()) {
    case VisItemKind::Module: {
      break;
    }
    case VisItemKind::ExternCrate: {
      break;
    }
    case VisItemKind::UseDeclaration: {
      break;
    }
    case VisItemKind::Function: {
      analyzeFunction(std::static_pointer_cast<Function>(visItem));
      break;
    }
    case VisItemKind::TypeAlias: {
      break;
    }
    case VisItemKind::Struct: {
      break;
    }
    case VisItemKind::Enumeration: {
      break;
    }
    case VisItemKind::Union: {
      break;
    }
    case VisItemKind::ConstantItem: {
      break;
    }
    case VisItemKind::StaticItem: {
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
}

} // namespace rust_compiler::sema
