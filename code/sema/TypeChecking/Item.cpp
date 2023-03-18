#include "AST/Function.h"
#include "AST/VisItem.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

void TypeResolver::checkVisItem(std::shared_ptr<ast::VisItem> v) {
  switch (v->getKind()) {
  case VisItemKind::Module: {
    assert(false && "to be implemented");
  }
  case VisItemKind::ExternCrate: {
    assert(false && "to be implemented");
  }
  case VisItemKind::UseDeclaration: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Function: {
    checkFunction(std::static_pointer_cast<Function>(v));
    break;
  }
  case VisItemKind::TypeAlias: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Struct: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Enumeration: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Union: {
    assert(false && "to be implemented");
  }
  case VisItemKind::ConstantItem: {
    assert(false && "to be implemented");
  }
  case VisItemKind::StaticItem: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Trait: {
    assert(false && "to be implemented");
  }
  case VisItemKind::Implementation: {
    assert(false && "to be implemented");
  }
  case VisItemKind::ExternBlock: {
    assert(false && "to be implemented");
  }
  }
}

void TypeResolver::checkMacroItem(std::shared_ptr<ast::MacroItem> v) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
