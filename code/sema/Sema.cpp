#include "Sema/Sema.h"

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/Module.h"
#include "AST/VisItem.h"

#include <llvm/Support/TimeProfiler.h>
#include <memory>

using namespace llvm;
using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Crate> &crate) {

  Sema sema;

  // path resolution
  // type checking
  // visiblity checks -> turns into linkage
  // match resp pattern exhaustive check
  // constant folding
  // drops

  sema.analyze(crate);
}

void Sema::analyze(std::shared_ptr<ast::Crate> &crate) {
  // FIXME
  { TimeTraceScope scope("name resolution"); }

  {
    TimeTraceScope scope("type inference");
    typeChecking.checkCrate(crate);
  }

  { TimeTraceScope scope("trait solving"); }

  { TimeTraceScope scope("visibility checks"); }

  { TimeTraceScope scope("exhaustive checks"); }

  { TimeTraceScope scope("drops"); }

  { TimeTraceScope scope("closure captures"); }

  for (const auto &item : crate->getItems()) {
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
