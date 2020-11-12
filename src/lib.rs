#![feature(const_ptr_offset)]

mod array;
mod number;
mod object;
mod string;
mod value;
pub use array::IArray;
pub use number::INumber;
pub use object::IObject;
pub use string::IString;
pub use value::IValue;
