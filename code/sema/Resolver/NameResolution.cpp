#include "NameResolution.h"

namespace rust_compiler::sema::resolver {

NameResolution::NameResolution(mappings::Mappings *_mappings, Resolver *_resolver) {
  mappings = _mappings;
  resolver = _resolver;

  // WHAT!

  // these are global
  resolver->getTypeScope().push(mappings->getNextNodeId());
  resolver->insertBuiltinTypes(resolver->getTypeScope().peek());
  resolver->pushNewTypeRib(resolver->getTypeScope().peek());
}

void NameResolution::resolve(std::shared_ptr<ast::Crate> crate) {
  // FIXME
  assert(false);
}

} // namespace rust_compiler::sema::resolver
