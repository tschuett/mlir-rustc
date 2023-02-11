#include "Resolver.h"

#include "ADT/CanonicalPath.h"

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

Rib *Scope::peek() { return stack.back(); }

void Scope::push(NodeId id) { stack.push_back(new Rib(getCrateNum(), id)); }

void Resolver::pushNewTypeRib(Rib *r) {
  if (typeRibs.size() == 0)
    globalTypeNodeId = r->getNodeId();

  assert(typeRibs.find(r->getNodeId()) == typeRibs.end());
  typeRibs[r->getNodeId()] = r;
}

void Resolver::insertBuiltinTypes(Rib *r) {
  auto builtins = getBuiltinTypes();
  for (auto &builtin : builtins) {
    CanonicalPath builtinPath =
        CanonicalPath::newSegment(builtin->getNodeId(), builtin->asString());
    r->insertName(builtinPath, builtin->getNodeId(),
                   Linemap::predeclared_location(), false, Rib::ItemType::Type,
                   [](const CanonicalPath &, NodeId, Location) -> void {});
  }
}

} // namespace rust_compiler::sema::resolver
