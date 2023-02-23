fn main() {
    Point { x: 10.0, y: 20.0 };
    NothingInMe {};
    TuplePoint(10.0, 20.0);
    TuplePoint { 0: 10.0, 1: 20.0 }; // Results in the same value as the above line
    let u = game::User {
        name: "Joe",
        age: 35,
        score: 100_000,
    };
    some_fn::<Cookie>(Cookie);

    struct Position(i32, i32, i32);
    Position(0, 0, 0);
    let c = Position;
    let pos = c(8, 6, 7);
}
