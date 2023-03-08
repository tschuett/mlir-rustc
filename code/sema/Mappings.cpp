#include "Sema/Mappings.h"

#include <memory>

using namespace rust_compiler::basic;

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

} // namespace rust_compiler::sema
