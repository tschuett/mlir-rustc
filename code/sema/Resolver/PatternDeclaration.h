#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "AST/Patterns/StructPattern.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "Resolver.h"

#include <map>
#include <string_view>

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

  bool isReference() const { return isRef; }

  Mutability getMutability() const { return mut; }
};

class PatternDeclaration {
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pat;
  RibKind rib;
  std::vector<PatternBinding> &bindings;

public:
  PatternDeclaration(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat,
                     RibKind rib, std::vector<PatternBinding> &bindings,
                     Resolver *resolver, adt::CanonicalPath prefix,
                     adt::CanonicalPath canonicalPrefix)
      : pat(pat), rib(rib), bindings(bindings), resolver(resolver),
        prefix(prefix), canonicalPrefix(canonicalPrefix) {}

  void resolve();

private:
  void resolvePattern(
      std::shared_ptr<ast::patterns::Pattern>);
  void resolvePatternWithoutRange(
      std::shared_ptr<ast::patterns::PatternWithoutRange>);
  void resolveIdentifierPattern(
      std::shared_ptr<ast::patterns::IdentifierPattern>);
  void resolveTupleStructPattern(
      std::shared_ptr<ast::patterns::TupleStructPattern>);
  void resolveStructPattern(
      std::shared_ptr<ast::patterns::StructPattern>);

  void addNewBinding(const lexer::Identifier &name, basic::NodeId id,
                     BindingTypeInfo bind);

  Resolver *resolver;

  adt::CanonicalPath prefix;
  adt::CanonicalPath canonicalPrefix;

  std::map<lexer::Identifier, BindingTypeInfo> bindingInfoMap;
};

} // namespace rust_compiler::sema::resolver
