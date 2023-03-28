#pragma once

#include "AST/Patterns/IdentifierPattern.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "Resolver.h"

#include <string_view>
#include <map>

namespace rust_compiler::sema::resolver {

enum class Mutability {
  Immutable,
  Mutable,
};

// Info that gets stored in the map. Helps us detect if two bindings to the same
// identifier have different mutability or ref states.
class BindingTypeInfo {
  Mutability mut;
  bool isRef;
  Location loc;

public:
  BindingTypeInfo(Mutability mut, bool isRef, Location loc)
      : mut(mut), isRef(isRef), loc(loc) {}

  Location getLocation() const { return loc; }
};

class PatternDeclaration {
  std::shared_ptr<ast::patterns::PatternWithoutRange> pat;
  RibKind rib;
  std::vector<PatternBinding> &bindings;

public:
  PatternDeclaration(std::shared_ptr<ast::patterns::PatternWithoutRange> pat,
                     RibKind rib, std::vector<PatternBinding> &bindings,
                     Resolver *resolver)
      : pat(pat), rib(rib), bindings(bindings), resolver(resolver) {}

  void resolve();

private:
  void resolvePatternWithoutRange(
      std::shared_ptr<ast::patterns::PatternWithoutRange>);
  void resolveIdentifierPattern(
      std::shared_ptr<ast::patterns::IdentifierPattern>);

  void addNewBinding(std::string_view name, basic::NodeId id,
                     BindingTypeInfo bind);

  Resolver *resolver;

  std::map<std::string, BindingTypeInfo> bindingInfoMap;
};

} // namespace rust_compiler::sema::resolver