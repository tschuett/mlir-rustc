#include "Sema/Mappings.h"

#include <memory>

using namespace rust_compiler::basic;
using namespace rust_compiler::ast;

namespace rust_compiler::sema {

Mappings *Mappings::get() {
  static std::unique_ptr<Mappings> instance;
  if (!instance)
    instance = std::unique_ptr<Mappings>(new Mappings());

  return instance.get();
}

NodeId Mappings::getNextNodeId() {
  auto it = nodeIdIter;
  ++nodeIdIter;
  return it;
}

void Mappings::insertModule(ast::Module *mod) { assert(false); }

ast::Module *Mappings::lookupModule(basic::NodeId id) {
  auto it = modules.find(id);
  if (it == modules.end())
    return nullptr;
  return it->second;
}

} // namespace rust_compiler::sema
