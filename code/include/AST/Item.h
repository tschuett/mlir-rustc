#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/OuterAttributes.h"
#include "AST/VisItem.h"
#include "Location.h"

#include <memory>

namespace rust_compiler::ast {

class Item : public Node {
  std::shared_ptr<OuterAttribute> outerAttributes;
  std::shared_ptr<VisItem> visItem;

public:
  explicit Item(rust_compiler::Location location) : Node(location) {}

  void setOuterAttributes(std::shared_ptr<OuterAttributes> outer);
  void setVisItem(std::shared_ptr<VisItem> visItem);

  std::shared_ptr<OuterAttributes> getOuterAttributes() const;

  std::shared_ptr<VisItem> getVisItem() const;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
