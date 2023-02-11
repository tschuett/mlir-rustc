#include "CrateLoader/Mappings.h"

#include <memory>

namespace rust_compiler::crate_loader {

Mappings::Mappings(){xxx

}

Mappings *Mappings::get() {
  static std::unique_ptr<Mappings> instance;

  if (!instance)
    instance = std::unique_ptr<Mappings>();

  return instance.get();
}

} // namespace rust_compiler::crate_loader
