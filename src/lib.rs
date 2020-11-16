#[macro_use]
mod macros;

mod array;
mod number;
mod object;
mod string;
mod value;
pub use array::IArray;
pub use number::INumber;
pub use object::IObject;
pub use string::IString;
pub use value::{Destructured, DestructuredMut, DestructuredRef, IValue};

mod de;
mod ser;
pub use de::from_value;
pub use ser::to_value;

#[cfg(test)]
mod tests {
    use mockalloc::Mockalloc;
    use std::alloc::System;

    #[global_allocator]
    static ALLOCATOR: Mockalloc<System> = Mockalloc(System);
}
