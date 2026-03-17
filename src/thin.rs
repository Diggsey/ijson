use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[repr(transparent)]
pub struct ThinRef<'a, T> {
    ptr: NonNull<T>,
    phantom: PhantomData<&'a T>,
}

impl<T> ThinRef<'_, T> {
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Deref for ThinRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr() }
    }
}

impl<T> Copy for ThinRef<'_, T> {}
impl<T> Clone for ThinRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[repr(transparent)]
pub struct ThinMut<'a, T> {
    ptr: NonNull<T>,
    phantom: PhantomData<&'a mut T>,
}

impl<T> ThinMut<'_, T> {
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Deref for ThinMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: `ptr` must be valid
        unsafe { &*self.ptr() }
    }
}

impl<T> DerefMut for ThinMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: `ptr` must be valid
        unsafe { &mut *self.ptr_mut() }
    }
}

pub trait ThinRefExt<'a, T>: Deref<Target = T> {
    fn ptr(&self) -> *const T;
}

pub trait ThinMutExt<'a, T>: DerefMut<Target = T> + ThinRefExt<'a, T> + Sized {
    fn ptr_mut(&mut self) -> *mut T;
    fn reborrow<'b>(&'b mut self) -> ThinMut<'b, T>;
}

impl<'a, T> ThinRefExt<'a, T> for ThinRef<'a, T> {
    fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<'a, T> ThinRefExt<'a, T> for ThinMut<'a, T> {
    fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<'a, T> ThinMutExt<'a, T> for ThinMut<'a, T> {
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
    fn reborrow<'b>(&'b mut self) -> ThinMut<'b, T> {
        Self {
            ptr: self.ptr,
            phantom: self.phantom,
        }
    }
}
