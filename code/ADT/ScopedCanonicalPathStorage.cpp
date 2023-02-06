#include "ADT/ScopedCanonicalPathStorage.h"

namespace rust_compiler::adt {

ScopedCanonicalPathStorageScope::ScopedCanonicalPathStorageScope(
    ScopedCanonicalPathStorage *storage, std::string_view segment){

    path.append(segment)

};

ScopedCanonicalPathStorageScope::~ScopedCanonicalPathStorageScope() {}

CanonicalPath ScopedCanonicalPathStorage::getCurrentPath() const { xxx }

} // namespace rust_compiler::adt
