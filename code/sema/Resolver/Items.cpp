#include "Resolver.h"

namespace rust_compiler::sema::resolver {

void Resolver::resolveModule(std::shared_ptr<ast::Module>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveVisItem(std::shared_ptr<ast::VisItem>,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveStaticItem(std::shared_ptr<ast::StaticItem>,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveConstantItem(std::shared_ptr<ast::ConstantItem>,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
