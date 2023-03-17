#include "Sema/Sema.h"

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/Module.h"
#include "AST/VisItem.h"
#include "Resolver/Resolver.h"
#include "TypeChecking/TypeChecking.h"

#include <llvm/Support/TimeProfiler.h>
#include <memory>

using namespace llvm;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::sema::resolver;

namespace rust_compiler::sema {

void analyzeSemantics(std::shared_ptr<ast::Crate> &crate) {

  Sema sema;

  sema.analyze(crate);

  // path resolution
  // type checking
  // visiblity checks -> turns into linkage
  // match resp pattern exhaustive check
  // constant folding
  // drops

  // Trait resolution (AssociatedItems, Implementations, Traits, and Structs

  // monomorph
}

void Sema::analyze(std::shared_ptr<ast::Crate> &crate) {
  // FIXME: needs to be passed to CrateBuilder. Mappings knows everything
  Resolver resolver = {};

  {
    TimeTraceScope scope("name resolution");
    resolver.resolveCrate(crate);
  }

  { TimeTraceScope scope("type inference"); }

  { TimeTraceScope scope("trait solving"); }

  { TimeTraceScope scope("visibility checks"); }

  { TimeTraceScope scope("exhaustive checks"); }

  { TimeTraceScope scope("drops"); }

  { TimeTraceScope scope("closure captures"); }

  { TimeTraceScope scope("constant evaluation"); }

  for (const auto &item : crate->getItems()) {
    const auto &visItem = item->getVisItem();
    if (!(bool)visItem)
      continue;

    // FIXME: weird

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
