#pragma once

namespace rust_compiler::target_info {

class Type;

class Action {
public:
  enum class ActionKind {
    CopyToMemoryKind,
    SizeRoundUpKind,
    SetSizeWithUnspecifiedUpperKind,
    ZeroOrSignExtendKind
  };

private:
  const ActionKind kind;

public:
  Action(ActionKind K) : kind(K) {}
  ActionKind getKind() const { return kind; }
};

/// copy to memory and use pointer
class CopyToMemoryAction : public Action {
  Type *type;

public:
  CopyToMemoryAction(Type *type)
      : Action(ActionKind::CopyToMemoryKind), type(type) {}
};

/// rounded up to the nearest multiple of X bytes.
class SizeRoundUpAction : public Action {
  Type *type;
  unsigned multiple;

public:
  SizeRoundUpAction(Type *type, unsigned multiple)
      : Action(ActionKind::SizeRoundUpKind), type(type), multiple(multiple) {}
};

/// The effect is as if the argument had been copied to the least significant
/// bits of a 64 bit register and the remaining bits filled with unspecified
/// values. https://github.com/rust-lang/rust/issues/97463
class SetSizeWithUnspecifiedUpper : public Action {
  Type *type;
  unsigned size;

public:
  SetSizeWithUnspecifiedUpper(Type *type, unsigned size)
      : Action(ActionKind::SetSizeWithUnspecifiedUpperKind), type(type),
        size(size) {}
};

class ZeroOrSignExtendAction : public Action {
  Type *arg;
  unsigned size;

public:
  ZeroOrSignExtendAction(Type *arg, unsigned size)
      : Action(ActionKind::ZeroOrSignExtendKind), arg(arg), size(size) {}
};

} // namespace rust_compiler::target_info
