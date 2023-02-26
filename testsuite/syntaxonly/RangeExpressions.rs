fn foo() {
    1..2; // std::ops::Range
    3..; // std::ops::RangeFrom
    ..4; // std::ops::RangeTo
    ..; // std::ops::RangeFull
    5..=6; // std::ops::RangeInclusive
    ..=7; // std::ops::RangeToInclusive

    let x = std::ops::Range { start: 0, end: 10 };
    let y = 0..10;
}
