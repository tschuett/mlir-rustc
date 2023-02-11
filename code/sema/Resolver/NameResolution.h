#pragma once

#include "AST/Crate.h"
#include "Resolver.h"
#include "Sema/Sema.h"

#include "Mappings/Mappings.h"

#include <memory>

namespace rust_compiler::sema::resolver {

class NameResolution {
public:
  NameResolution(mappings::Mappings *mapping, Resolver * resolver);

  void resolve(std::shared_ptr<ast::Crate> crate);

private:
  mappings::Mappings *mappings;
  Resolver *resolver;
};

} // namespace rust_compiler::sema::resolver
