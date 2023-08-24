#include "Sema/Rib.h"

using namespace rust_compiler::sema;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;

StringResult<NodeId> Rib::insert(std::string_view name,
                                             NodeId id, bool canShadow) {
  auto res = values.insert({std::string(name), id});
  auto insertedId = res.first->second;
  bool existed = !res.second;

  if (existed && !canShadow)
    return StringResult<basic::NodeId>(std::string("shadowed insert"));

  return StringResult<NodeId>(insertedId);
}


std::optional<NodeId> Rib::get(std::string_view name) {
  auto it = values.find(std::string(name));
  if (it == values.end())
    return std::nullopt;

  return it->second;
}
