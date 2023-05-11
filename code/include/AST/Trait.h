#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/Types.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"
#include "Location.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class Trait : public VisItem {
  bool unsafe;
  Identifier identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItems;

public:
  Trait(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Trait, vis) {}

  void setUnsafe() { unsafe = true; }
  void setIdentifier(const Identifier &id) { identifier = id; }
  void setGenericParams(const GenericParams &p) { genericParams = p; }

  void setBounds(const types::TypeParamBounds &b) { typeParamBounds = b; }
  void setWhere(const WhereClause &w) { whereClause = w; }

  void setInner(std::vector<InnerAttribute> &inn) { innerAttributes = inn; }

  void addItem(const AssociatedItem &item) { associatedItems.push_back(item); }

  Identifier getIdentifier() const { return identifier; }

  bool hasGenericParams() const { return genericParams.has_value(); }
  GenericParams getGenericParams() const { return *genericParams; }
  bool hasWhereClause() const { return whereClause.has_value(); }
  WhereClause getWhereClause() const { return *whereClause; }
  bool hasTypeParamBounds() const { return typeParamBounds.has_value(); }
  types::TypeParamBounds getTypeParamBounds() const { return *typeParamBounds; }
  std::vector<AssociatedItem> getAssociatedItems() const {
    return associatedItems;
  }

  // convinience
  void insertImplicitSelf(const GenericParam &gp) {
    if (genericParams) {
      GenericParams tmp = *genericParams;
      tmp.addGenericParam(gp);
      genericParams = tmp;
      return;
    }
    GenericParams tmp = {getLocation()};
    tmp.addGenericParam(gp);
    genericParams = tmp;
  }
};

} // namespace rust_compiler::ast
