use rkyv::{ser::Serializer, Serialize};
use rkyv::{Archive, Archived, Deserialize, Fallible};
use serde::Deserializer;

use super::array::IArray;
use super::number::INumber;
use super::object::IObject;
use super::value::IValue;

impl<S: Serializer> Serialize<S> for IValue {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        todo!()
    }
}

impl<S: Serializer> Serialize<S> for INumber {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        todo!()
    }
}

impl<S: Serializer> Serialize<S> for IArray {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        todo!()
    }
}

impl<S: Serializer> Serialize<S> for IObject {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        todo!()
    }
}

impl Archive for IValue {
    type Archived = IValue;

    type Resolver = ();

    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        todo!()
    }
}

impl Archive for INumber {
    type Archived = INumber;

    type Resolver = ();

    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        todo!()
    }
}

impl Archive for IArray {
    type Archived = IArray;

    type Resolver = ();

    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        todo!()
    }
}

impl Archive for IObject {
    type Archived = IObject;

    type Resolver = ();

    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        todo!()
    }
}

impl<D: Fallible + ?Sized> Deserialize<IValue, D> for Archived<IValue> {
    fn deserialize(&self, deserializer: &mut D) -> Result<IValue, D::Error> {
        todo!()
    }
}

impl<D: Fallible + ?Sized> Deserialize<INumber, D> for Archived<INumber> {
    fn deserialize(&self, deserializer: &mut D) -> Result<INumber, D::Error> {
        todo!()
    }
}
