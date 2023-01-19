#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h> // for debugging
#include <optional>
#include <string>

namespace rust_compiler::adt {

template <typename K, typename V, typename KInfo = llvm::DenseMapInfo<K>>
class SmallMap {
  llvm::DenseMap<K, V, KInfo> localMap;

public:
  //   SmallMap() {
  //     llvm::outs() << "  SmallMap()"
  //                  << "\n";
  //   }

  //   ~SmallMap() {
  //     llvm::outs() << "  ~SmallMap()"
  //                  << "\n";
  //   }

  [[nodiscard]] bool contains(const K &key) const {
    return localMap.count(key) == 1;
  }

  [[nodiscard]] bool insert(const K &key, const V &value) {
    auto res = localMap.insert({key, value});
    return std::get<1>(res);
  }

  [[nodiscard]] std::optional<V> find(const K &Key) const {
    auto it = localMap.find(Key);
    if (it != localMap.end())
      return it->second;
    return std::nullopt;
  }
};

template <typename K, typename V, typename KInfo> class ScopedHashTable;

template <typename K, typename V, typename KInfo = llvm::DenseMapInfo<K>>
class ScopedHashTableScope {
  /// HT - The hashtable that we are active for.
  ScopedHashTable<K, V, KInfo> &HT;

  /// PreviousScope - This is the scope that we are shadowing in HT.
  ScopedHashTableScope *previousScope = nullptr;

  SmallMap<K, V> localMap;

public:
  ScopedHashTableScope(ScopedHashTable<K, V, KInfo> &HT) : HT(HT) {
    // assert(HT.currentScope != nullptr && "No active scope");
    previousScope = HT.currentScope;
    HT.currentScope = this;
  }

  ~ScopedHashTableScope() {
    // FIXME
    HT.currentScope = previousScope;
  }

  [[nodiscard]] bool contains(const K &key) const {
    if (localMap.contains(key)) {
      return true;
    }
    if (previousScope == nullptr) {
      return false;
    }
    return previousScope->contains(key);
  }

  [[nodiscard]] bool insert(const K &key, const V &value) {
    return localMap.insert(key, value);
  }

  [[nodiscard]] std::optional<V> find(const K &Key) const {
    if (localMap.contains(Key))
      return localMap.find(Key);
    if (previousScope == nullptr)
      return std::nullopt;
    return previousScope->find(Key);
  }
};

template <typename K, typename V, typename KInfo = llvm::DenseMapInfo<K>>
class ScopedHashTable {

  /// ScopeTy - This is a helpful typedef that allows clients to get easy
  /// access to the name of the scope for this hash table.
  using ScopeTy = ScopedHashTableScope<K, V, KInfo>;

  ScopeTy *currentScope = nullptr;

  friend class ScopedHashTableScope<K, V, KInfo>;

public:
  ScopedHashTable() = default;

  [[nodiscard]] bool contains(const K &key) const {
    assert(currentScope != nullptr && "No active scope");
    return currentScope->contains(key);
  }

  bool insert(const K &key, const V &value) {
    assert(currentScope != nullptr && "No active scope");
    return currentScope->insert(key, value);
  }

  [[nodiscard]] std::optional<V> find(const K &Key) const {
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

template <> struct std::hash<llvm::StringRef> {
  std::size_t operator()(const llvm::StringRef &ref) const {
    using std::hash;
    using std::size_t;
    using std::string;

    std::string s = std::string(ref);

    return std::hash<std::string>{}(s);
  }
};

// FIXME: slow
