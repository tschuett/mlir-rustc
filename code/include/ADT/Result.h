#pragma once

#include <cassert>
#include <optional>
#include <variant>

namespace rust_compiler::adt {

/// A Result type for error handling.
/// Inspired by https://doc.rust-lang.org/std/result/enum.Result.html

/// Note that std::optional is a cheap trick
template <typename T, typename E> class Result {
  std::variant<std::optional<T>, E> storage;

  bool checked = false;

public:
  static_assert(!std::is_same_v<T, E>);

  explicit Result(const T &t) { storage.template emplace<0>(t); }
  explicit Result(const E &e) { storage.template emplace<1>(e); }

  ~Result() {
    //    if (not checked) {
    //      assert(false);
    //    }
  }

  /// Return false if there is an error.
  operator bool() { return isOk(); }

  /// Can throw
  T getValue() { return *std::get<std::optional<T>>(storage); }
  /// Can throw
  E getError() { return std::get<E>(storage); }

  T &&moveValue() { return std::move(*std::get<std::optional<T>>(storage)); }

  bool isOk() noexcept {
    checked = true;
    return std::holds_alternative<std::optional<T>>(storage);
  }

  bool isErr() noexcept {
    checked = true;
    return std::holds_alternative<E>(storage);
  }
};

template <class T> using StringResult = Result<T, std::string>;

} // namespace rust_compiler::adt
