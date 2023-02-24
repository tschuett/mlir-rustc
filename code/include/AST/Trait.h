#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Expression.h"
#include "AST/GenericParams.h"
#include "AST/InnerAttribute.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/Types.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class Trait : public VisItem {
  bool unsafe;
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<types::TypeParamBounds> typeParamBounds;
  std::optional<WhereClause> whereClause;

  std::vector<InnerAttribute> innerAttributes;
  std::vector<AssociatedItem> associatedItem;

public:
  Trait(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Trait, vis) {}

  std::span<std::shared_ptr<AssociatedItem>> getAssociatedItems() const;

  void setUnsafe() { unsafe = true; }
  void setIdentifier(std::string_view id) { identifier = id; }
  void setGenericParams(const GenericParams &p) { genericParams = p; }

  void setBounds(const types::TypeParamBounds &b) { typeParamBounds = b; }
  void setWhere(const WhereClause &w) { whereClause = w; }

  void setInner(std::vector<InnerAttribute> &inn) { innerAttributes = inn; }

  void addItem(const AssociatedItem& item) {
    associatedItem.push_back(item);
  }
};

} // namespace rust_compiler::ast
