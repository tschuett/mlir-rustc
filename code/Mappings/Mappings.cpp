#include "Mappings/Mappings.h"

#include <memory>

namespace rust_compiler::mappings {

Mappings *Mappings::get() {
  static std::unique_ptr<Mappings> instance;
  if (!instance)
    instance = std::unique_ptr<Mappings>(new Mappings());

  return instance.get();
}

} // namespace rust_compiler::mappings
