pub enum Option2<T> {
    None,
    Some(T),
}

struct Person {
    car: Option2<u32>,
    age: u32,
    name: u32,
    next: u32,
}

fn main() {
    enum E {
        X(u8),
        Y(u8),
        Z(u8),
    }
    let v = E::Y(12);
    if let E::X(n) | E::Y(n) = v {
        if n == 12 {}
    }
    let person = Person {
        car: Option2::Some(5),
        age: 14,
        name: 5,
        next: 1,
    };
    if let Person {
        car: Option2::Some(_),
        age: person_age @ 13..=19,
        name: ref person_name,
        ..
    } = person
    {
        println!("{} has a car and is {} years old.", person_name, person_age);
    }

    enum Animal {
        Dog,
        Cat,
    }

    let mut a: Animal = Animal::Dog;
    a = Animal::Cat;

    #[repr(u8)]
    enum Enum2 {
        Unit = 3,
        Tuple(u16),
        Struct { a: u8, b: u16 } = 1,
    }

    enum ZeroVariants {}
}
