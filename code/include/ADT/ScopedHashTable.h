#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <optional>
#include <string>

namespace rust_compiler::adt {

template <typename K, typename V, typename KInfo> class ScopedHashTable;

template <typename K, typename V, typename KInfo = llvm::DenseMapInfo<K>>
class ScopedHashTableScope {
  /// HT - The hashtable that we are active for.
  ScopedHashTable<K, V, KInfo> &HT;

  llvm::DenseMap<K, V, KInfo> localMap;

  /// PreviousScope - This is the scope that we are shadowing in HT.
  ScopedHashTableScope *previousScope = nullptr;

public:
  ScopedHashTableScope(ScopedHashTable<K, V, KInfo> &HT) : HT(HT) {
    //assert(HT.currentScope != nullptr && "No active scope");
    previousScope = HT.currentScope;
    HT.currentScope = this;
  }

  ~ScopedHashTableScope() {
    // FIXME
    HT.currentScope = previousScope;
  }

  bool contains(const K &key) const {
    if (localMap.count(key) == 1)
      return true;
    if (previousScope == nullptr)
      return false;
    return previousScope->contains(key);
  }

  void insert(const K &key, const V &value) { localMap.insert({key, value}); }

  std::optional<V> find(const K &Key) const {
    if (localMap.count(Key) == 1)
      return localMap.lookup(Key);
    if (previousScope == nullptr)
      return std::nullopt;
    return previousScope->find(Key);
  }
};

template <typename K, typename V, typename KInfo = llvm::DenseMapInfo<K>>
class ScopedHashTable {

  /// ScopeTy - This is a helpful typedef that allows clients to get easy access
  /// to the name of the scope for this hash table.
  using ScopeTy = ScopedHashTableScope<K, V, KInfo>;

  ScopeTy *currentScope = nullptr;

  friend class ScopedHashTableScope<K, V, KInfo>;

public:
  ScopedHashTable() = default;

  bool contains(const K &key) const {
    assert(currentScope != nullptr && "No active scope");
    return currentScope->contains(key);
  }

  void insert(const K &key, const V &value) {
    assert(currentScope != nullptr && "No active scope");
    currentScope->insert(key, value);
  }

  std::optional<V> find(const K &Key) const {
    assert(currentScope != nullptr && "No active scope");
    return currentScope->find(Key);
  }
};

} // namespace rust_compiler::adt

namespace llvm {
template <> struct DenseMapInfo<std::string, void> {
  static inline std::string getEmptyKey() { return std::string(""); }

  static inline std::string getTombstoneKey() {
    return std::string("1234567890");
  }

  static unsigned getHashValue(std::string value) {
    return std::hash<std::string>{}(value);
  }

  static bool isEqual(std::string LHS, std::string RHS) {
    if (RHS.data() == getEmptyKey().data())
      return LHS.data() == getEmptyKey().data();
    if (RHS.data() == getTombstoneKey().data())
      return LHS.data() == getTombstoneKey().data();
    return LHS == RHS;
  }
};

} // namespace llvm

// FIXME: slow
