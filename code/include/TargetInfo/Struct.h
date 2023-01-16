#pragma once

#include "TargetInfo/Types.h"

#include <variant>

namespace rust_compiler::target_info {

class StructType : public Type {
  std::vector<std::variant<Type *, unsigned>> members;

  size_t sizeInBytes;
  size_t alignmentInBytes;

public:
  StructType(std::vector<std::variant<Type *, unsigned>> members,
             size_t sizeInBytes, size_t alignmentInBytes)
      : Type(TypeKind::StructType), members(members), sizeInBytes(sizeInBytes),
        alignmentInBytes(alignmentInBytes) {}

  std::vector<Type *> getMembers() const {
    std::vector<Type *> mems;

    for (auto var : members) {
      if (std::holds_alternative<Type *>(var)) {
        mems.push_back(std::get<Type *>(var));
      }
    }

    return mems;
  }

  size_t getSizeInBytes() const { return sizeInBytes; }
  size_t getAlignmentInBytes() const { return alignmentInBytes; }

  static bool classof(const Type *S) {
    return S->getKind() == TypeKind::StructType;
  }
};

} // namespace rust_compiler::target_info
