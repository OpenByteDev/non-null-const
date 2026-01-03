#![no_std]
#![allow(internal_features)]
#![feature(
    sized_hierarchy,
    negative_impls,
    const_trait_impl,
    pin_coerce_unsized_trait,
    unsize,
    const_destruct,
    const_option_ops,
    intra_doc_pointers,
    const_convert
)]
#![cfg_attr(feature = "ptr_as_uninit", feature(ptr_as_uninit))]
#![cfg_attr(feature = "ptr_cast_array", feature(ptr_cast_array))]
#![cfg_attr(feature = "ptr_metadata", feature(ptr_metadata))]
#![cfg_attr(
    feature = "pointer_try_cast_aligned",
    feature(pointer_try_cast_aligned)
)]
#![cfg_attr(feature = "const_drop_in_place", feature(const_drop_in_place))]
#![cfg_attr(feature = "pointer_is_aligned_to", feature(pointer_is_aligned_to))]
#![cfg_attr(feature = "cast_maybe_uninit", feature(cast_maybe_uninit))]
#![cfg_attr(feature = "slice_ptr_get", feature(slice_ptr_get))]
#![cfg_attr(feature = "coerce_unsized", feature(coerce_unsized))]
#![cfg_attr(feature = "dispatch_from_dyn", feature(dispatch_from_dyn))]
#![cfg_attr(feature = "ptr_internals", feature(ptr_internals))]
#![cfg_attr(feature = "const_index", feature(const_index))]

//! This crate provides [`NonNullConst`], a **non-null, covariant** raw **const** pointer type. It is conceptually the `*const` analogue of [`core::ptr::NonNull`].

#[cfg(any(feature = "slice_ptr_get", feature = "ptr_internals"))]
use cfg_tt::cfg_tt;

#[cfg(any(feature = "coerce_unsized", feature = "dispatch_from_dyn"))]
use core::marker::Unsize;
#[cfg(feature = "coerce_unsized")]
use core::ops::CoerceUnsized;
#[cfg(feature = "dispatch_from_dyn")]
use core::ops::DispatchFromDyn;
#[cfg(feature = "ptr_metadata")]
use core::ptr;
#[cfg(feature = "ptr_internals")]
use core::ptr::Unique;
#[cfg(any(feature = "ptr_internals", feature = "slice_ptr_get"))]
use core::slice::SliceIndex;
use core::{
    cmp::Ordering, fmt, hash, marker::PointeeSized, mem::MaybeUninit, num::NonZero,
    pin::PinCoerceUnsized,
};

pub use core::ptr::NonNull as NonNullMut;

/// `*const T` but non-zero and [covariant].
///
/// This is often the correct thing to use when building data structures using
/// raw pointers, but is ultimately more dangerous to use because of its additional
/// properties. If you're not sure if you should use `NonNullConst<T>`, just use `*const T`!
///
/// Unlike `*const T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<NonNullConst<T>>` has the same size as `*const T`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*const T`, `NonNullConst<T>` is covariant over `T`. This is usually the correct
/// choice for most data structures and safe abstractions, such as `Box`, `Rc`, `Arc`, `Vec`,
/// and `LinkedList`.
///
/// # Representation
///
/// Thanks to the [null pointer optimization],
/// `NonNullConst<T>` and `Option<NonNullConst<T>>`
/// are guaranteed to have the same size and alignment:
///
/// ```
/// use non_null_const::NonNullConst;
///
/// assert_eq!(size_of::<NonNullConst<i16>>(), size_of::<Option<NonNullConst<i16>>>());
/// assert_eq!(align_of::<NonNullConst<i16>>(), align_of::<Option<NonNullConst<i16>>>());
///
/// assert_eq!(size_of::<NonNullConst<str>>(), size_of::<Option<NonNullConst<str>>>());
/// assert_eq!(align_of::<NonNullConst<str>>(), align_of::<Option<NonNullConst<str>>>());
/// ```
///
/// [covariant]: https://doc.rust-lang.org/reference/subtyping.html
/// [null pointer optimization]: core::option#representation
#[repr(transparent)]
pub struct NonNullConst<T: PointeeSized>(NonNullMut<T>);

/// `NonNullConst` pointers are not `Send` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
impl<T: PointeeSized> !Send for NonNullConst<T> {}

/// `NonNullConst` pointers are not `Sync` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
impl<T: PointeeSized> !Sync for NonNullConst<T> {}

impl<T: Sized> NonNullConst<T> {
    /// Creates a pointer with the given address and no [provenance][core::ptr#provenance].
    ///
    /// For more details, see the equivalent method on a raw pointer, [`ptr::without_provenance`].
    ///
    /// This is a [Strict Provenance][core::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    pub const fn without_provenance(addr: NonZero<usize>) -> Self {
        Self(NonNullMut::without_provenance(addr))
    }

    /// Creates a new `NonNullConst` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the address of the returned pointer may potentially
    /// be that of a valid pointer, which means this must not be used
    /// as a "not yet initialized" sentinel value.
    /// Types that lazily allocate must track initialization by some other means.
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let ptr = NonNullConst::<u32>::dangling();
    /// // Important: don't try to access the value of `ptr` without
    /// // initializing it first! The pointer is not null but isn't valid either!
    /// ```
    #[must_use]
    #[inline]
    pub const fn dangling() -> Self {
        Self(NonNullMut::dangling())
    }

    /// Converts an address back to a immutable pointer, picking up some previously 'exposed'
    /// [provenance][core::ptr#provenance].
    ///
    /// For more details, see the equivalent method on a raw pointer, [`ptr::with_exposed_provenance`].
    ///
    /// This is an [Exposed Provenance][core::ptr#exposed-provenance] API.
    #[inline]
    pub fn with_exposed_provenance(addr: NonZero<usize>) -> Self {
        Self(NonNullMut::with_exposed_provenance(addr))
    }

    /// Returns a shared references to the value. In contrast to [`as_ref`], this does not require
    /// that the value has to be initialized.
    ///
    /// [`as_ref`]: NonNullConst::as_ref
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](core::ptr#pointer-to-reference-conversion).
    /// Note that because the created reference is to `MaybeUninit<T>`, the
    /// source pointer can point to uninitialized memory.
    #[inline]
    #[must_use]
    #[cfg(feature = "ptr_as_uninit")]
    pub const unsafe fn as_uninit_ref<'a>(self) -> &'a MaybeUninit<T> {
        unsafe { self.0.as_uninit_ref() }
    }

    /// Casts from a pointer-to-`T` to a pointer-to-`[T; N]`.
    #[inline]
    #[cfg(feature = "ptr_cast_array")]
    pub const fn cast_array<const N: usize>(self) -> NonNullConst<[T; N]> {
        NonNullConst(self.0.cast())
    }
}

impl<T: PointeeSized> NonNullConst<T> {
    /// Creates a new `NonNullConst`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u32;
    /// let ptr = unsafe { NonNullConst::new_unchecked(&x as *const _) };
    /// ```
    ///
    /// *Incorrect* usage of this function:
    ///
    /// ```rust,no_run
    /// use non_null_const::NonNullConst;
    ///
    /// // NEVER DO THAT!!! This is undefined behavior. ⚠️
    /// let ptr = unsafe { NonNullConst::<u32>::new_unchecked(std::ptr::null()) };
    /// ```
    #[inline]
    #[track_caller]
    pub const unsafe fn new_unchecked(ptr: *const T) -> Self {
        unsafe { Self(NonNullMut::new_unchecked(ptr as *mut _)) }
    }

    /// Creates a new `NonNullConst` if `ptr` is non-null.
    ///
    /// # Panics during const evaluation
    ///
    /// This method will panic during const evaluation if the pointer cannot be
    /// determined to be null or not. See [`is_null`] for more information.
    ///
    /// [`is_null`]: ../primitive.pointer.html#method.is_null-1
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u32;
    /// let ptr = NonNullConst::<u32>::new(&x as *const _).expect("ptr is null!");
    ///
    /// if let Some(ptr) = NonNullConst::<u32>::new(std::ptr::null()) {
    ///     unreachable!();
    /// }
    /// ```
    #[inline]
    pub const fn new(ptr: *const T) -> Option<Self> {
        NonNullMut::new(ptr as *mut _).map(Self)
    }

    /// Converts a reference to a `NonNullConst` pointer.
    #[inline]
    pub const fn from_ref(r: &T) -> Self {
        Self(NonNullMut::from_ref(r))
    }

    /// Converts a mutable reference to a `NonNullConst` pointer.
    #[inline]
    pub const fn from_mut(r: &mut T) -> Self {
        Self(NonNullMut::from_mut(r))
    }

    /// Performs the same functionality as [`std::ptr::from_raw_parts`], except that a
    /// `NonNullConst` pointer is returned, as opposed to a raw `*const` pointer.
    ///
    /// See the documentation of [`std::ptr::from_raw_parts`] for more details.
    ///
    /// [`std::ptr::from_raw_parts`]: core::ptr::from_raw_parts
    #[cfg(feature = "ptr_metadata")]
    #[inline]
    pub const fn from_raw_parts(
        data_pointer: NonNullConst<impl ptr::Thin>,
        metadata: <T as ptr::Pointee>::Metadata,
    ) -> NonNullConst<T> {
        Self(NonNullMut::from_raw_parts(data_pointer.0, metadata))
    }

    /// Decompose a (possibly wide) pointer into its data pointer and metadata components.
    ///
    /// The pointer can be later reconstructed with [`NonNullConst::from_raw_parts`].
    #[cfg(feature = "ptr_metadata")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_raw_parts(self) -> (NonNullConst<()>, <T as ptr::Pointee>::Metadata) {
        let (data_pointer, metadata) = self.0.to_raw_parts();
        (NonNullConst::<()>(data_pointer), metadata)
    }

    /// Gets the "address" portion of the pointer.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::addr`].
    ///
    /// This is a [Strict Provenance][core::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    pub fn addr(self) -> NonZero<usize> {
        self.0.addr()
    }

    /// Exposes the ["provenance"][core::ptr#provenance] part of the pointer for future use in
    /// [`with_exposed_provenance`][NonNullConst::with_exposed_provenance] and returns the "address" portion.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::expose_provenance`].
    ///
    /// This is an [Exposed Provenance][core::ptr#exposed-provenance] API.
    pub fn expose_provenance(self) -> NonZero<usize> {
        self.0.expose_provenance()
    }

    /// Creates a new pointer with the given address and the [provenance][core::ptr#provenance] of
    /// `self`.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::with_addr`].
    ///
    /// This is a [Strict Provenance][core::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    pub fn with_addr(self, addr: NonZero<usize>) -> Self {
        Self(self.0.with_addr(addr))
    }

    /// Creates a new pointer by mapping `self`'s address to a new one, preserving the
    /// [provenance][core::ptr#provenance] of `self`.
    ///
    /// For more details, see the equivalent method on a raw pointer, [`pointer::map_addr`].
    ///
    /// This is a [Strict Provenance][core::ptr#strict-provenance] API.
    #[must_use]
    #[inline]
    pub fn map_addr(self, f: impl FnOnce(NonZero<usize>) -> NonZero<usize>) -> Self {
        Self(self.0.map_addr(f))
    }

    /// Acquires the underlying `*const` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u32;
    /// let ptr = NonNullConst::new(&x).expect("ptr is null!");
    ///
    /// let x_value = unsafe { *ptr.as_ptr() };
    /// assert_eq!(x_value, 0);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(self) -> *const T {
        self.0.as_ptr()
    }

    /// Returns a shared reference to the value. If the value may be uninitialized, [`as_uninit_ref`]
    /// must be used instead.
    ///
    /// [`as_uninit_ref`]: NonNullConst::as_uninit_ref
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that
    /// the pointer is [convertible to a reference](core::ptr#pointer-to-reference-conversion).
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u32;
    /// let ptr = NonNullConst::new(&x as *const _).expect("ptr is null!");
    ///
    /// let ref_x = unsafe { ptr.as_ref() };
    /// println!("{ref_x}");
    /// ```
    ///
    /// [the module documentation]: core::ptr#safety
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_ref<'a>(&self) -> &'a T {
        unsafe { self.0.as_ref() }
    }

    /// Casts to a pointer of another type.
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u32;
    /// let ptr = NonNullConst::new(&x as *const _).expect("null pointer");
    ///
    /// let casted_ptr = ptr.cast::<i8>();
    /// let raw_ptr: *const i8 = casted_ptr.as_ptr();
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn cast<U>(self) -> NonNullConst<U> {
        NonNullConst(self.0.cast())
    }

    /// Try to cast to a pointer of another type by checking alignment.
    ///
    /// If the pointer is properly aligned to the target type, it will be
    /// cast to the target type. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(pointer_try_cast_aligned)]
    /// use non_null_const::NonNullConst;
    ///
    /// let x = 0u64;
    ///
    /// let aligned = NonNullConst::from_ref(&x);
    /// let unaligned = unsafe { aligned.byte_add(1) };
    ///
    /// assert!(aligned.try_cast_aligned::<u32>().is_some());
    /// assert!(unaligned.try_cast_aligned::<u32>().is_none());
    /// ```
    #[cfg(feature = "pointer_try_cast_aligned")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn try_cast_aligned<U>(self) -> Option<NonNullConst<U>> {
        self.0.try_cast_aligned().map(NonNullConst::<U>)
    }

    /// Adds an offset to a pointer.
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: core::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let s = [1, 2, 3];
    /// let ptr: NonNullConst<u32> = NonNullConst::new(s.as_ptr()).unwrap();
    ///
    /// unsafe {
    ///     println!("{}", ptr.offset(1).read());
    ///     println!("{}", ptr.offset(2).read());
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    pub const unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        unsafe { Self(self.0.offset(count)) }
    }

    /// Calculates the offset from a pointer in bytes.
    ///
    /// `count` is in units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [offset][pointer::offset] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn byte_offset(self, count: isize) -> Self {
        unsafe { Self(self.0.byte_offset(count)) }
    }

    /// Adds an offset to a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: core::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let s: &str = "123";
    /// let ptr: NonNullConst<u8> = NonNullConst::new(s.as_ptr().cast()).unwrap();
    ///
    /// unsafe {
    ///     println!("{}", ptr.add(1).read() as char);
    ///     println!("{}", ptr.add(2).read() as char);
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    pub const unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        unsafe { Self(self.0.add(count)) }
    }

    /// Calculates the offset from a pointer in bytes (convenience for `.byte_offset(count as isize)`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`add`][NonNullConst::add] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn byte_add(self, count: usize) -> Self {
        unsafe { Self(self.0.byte_add(count)) }
    }

    /// Subtracts an offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g., a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * The computed offset, `count * size_of::<T>()` bytes, must not overflow `isize`.
    ///
    /// * If the computed offset is non-zero, then `self` must be derived from a pointer to some
    ///   [allocation], and the entire memory range between `self` and the result must be in
    ///   bounds of that allocation. In particular, this range must not "wrap around" the edge
    ///   of the address space.
    ///
    /// Allocations can never be larger than `isize::MAX` bytes, so if the computed offset
    /// stays in bounds of the allocation, it is guaranteed to satisfy the first requirement.
    /// This implies, for instance, that `vec.as_ptr().add(vec.len())` (for `vec: Vec<T>`) is always
    /// safe.
    ///
    /// [allocation]: core::ptr#allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let s: &str = "123";
    ///
    /// unsafe {
    ///     let end: NonNullConst<u8> = NonNullConst::new(s.as_ptr().cast()).unwrap().add(3);
    ///     println!("{}", end.sub(1).read() as char);
    ///     println!("{}", end.sub(2).read() as char);
    /// }
    /// ```
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[must_use = "returns a new pointer rather than modifying its argument"]
    pub const unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        unsafe { Self(self.0.sub(count)) }
    }

    /// Calculates the offset from a pointer in bytes (convenience for
    /// `.byte_offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of bytes.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`sub`][NonNullConst::sub] on it. See that method for documentation
    /// and safety requirements.
    ///
    /// For non-`Sized` pointees this operation changes only the data pointer,
    /// leaving the metadata untouched.
    #[must_use]
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn byte_sub(self, count: usize) -> Self {
        unsafe { Self(self.0.byte_sub(count)) }
    }

    /// Calculates the distance between two pointers within the same allocation. The returned value is in
    /// units of T: the distance in bytes divided by `size_of::<T>()`.
    ///
    /// This is equivalent to `(self as isize - origin as isize) / (size_of::<T>() as isize)`,
    /// except that it has a lot more opportunities for UB, in exchange for the compiler
    /// better understanding what you are doing.
    ///
    /// The primary motivation of this method is for computing the `len` of an array/slice
    /// of `T` that you are currently representing as a "start" and "end" pointer
    /// (and "end" is "one past the end" of the array).
    /// In that case, `end.offset_from(start)` gets you the length of the array.
    ///
    /// All of the following safety requirements are trivially satisfied for this usecase.
    ///
    /// [`offset`]: #method.offset
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined Behavior:
    ///
    /// * `self` and `origin` must either
    ///
    ///   * point to the same address, or
    ///   * both be *derived from* a pointer to the same [allocation], and the memory range between
    ///     the two pointers must be in bounds of that object. (See below for an example.)
    ///
    /// * The distance between the pointers, in bytes, must be an exact multiple
    ///   of the size of `T`.
    ///
    /// As a consequence, the absolute distance between the pointers, in bytes, computed on
    /// mathematical integers (without "wrapping around"), cannot overflow an `isize`. This is
    /// implied by the in-bounds requirement, and the fact that no allocation can be larger
    /// than `isize::MAX` bytes.
    ///
    /// The requirement for pointers to be derived from the same allocation is primarily
    /// needed for `const`-compatibility: the distance between pointers into *different* allocated
    /// objects is not known at compile-time. However, the requirement also exists at
    /// runtime and may be exploited by optimizations. If you wish to compute the difference between
    /// pointers that are not guaranteed to be from the same allocation, use `(self as isize -
    /// origin as isize) / size_of::<T>()`.
    // FIXME: recommend `addr()` instead of `as usize` once that is stable.
    ///
    /// [`add`]: #method.add
    /// [allocation]: core::ptr#allocation
    ///
    /// # Panics
    ///
    /// This function panics if `T` is a Zero-Sized Type ("ZST").
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let a = [0; 5];
    /// let ptr1: NonNullConst<u32> = NonNullConst::from(&a[1]);
    /// let ptr2: NonNullConst<u32> = NonNullConst::from(&a[3]);
    /// unsafe {
    ///     assert_eq!(ptr2.offset_from(ptr1), 2);
    ///     assert_eq!(ptr1.offset_from(ptr2), -2);
    ///     assert_eq!(ptr1.offset(2), ptr2);
    ///     assert_eq!(ptr2.offset(-2), ptr1);
    /// }
    /// ```
    ///
    /// *Incorrect* usage:
    ///
    /// ```rust,no_run
    /// use non_null_const::NonNullConst;
    ///
    /// let ptr1 = NonNullConst::new(Box::into_raw(Box::new(0u8))).unwrap();
    /// let ptr2 = NonNullConst::new(Box::into_raw(Box::new(1u8))).unwrap();
    /// let diff = (ptr2.addr().get() as isize).wrapping_sub(ptr1.addr().get() as isize);
    /// // Make ptr2_other an "alias" of ptr2.add(1), but derived from ptr1.
    /// let diff_plus_1 = diff.wrapping_add(1);
    /// let ptr2_other = NonNullConst::new(ptr1.as_ptr().wrapping_byte_offset(diff_plus_1)).unwrap();
    /// assert_eq!(ptr2.addr(), ptr2_other.addr());
    /// // Since ptr2_other and ptr2 are derived from pointers to different objects,
    /// // computing their offset is undefined behavior, even though
    /// // they point to addresses that are in-bounds of the same object!
    ///
    /// let one = unsafe { ptr2_other.offset_from(ptr2) }; // Undefined Behavior! ⚠️
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset_from(self, origin: NonNullConst<T>) -> isize
    where
        T: Sized,
    {
        unsafe { self.0.offset_from(origin.0) }
    }

    /// Calculates the distance between two pointers within the same allocation. The returned value is in
    /// units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`offset_from`][NonNullConst::offset_from] on it. See that method for
    /// documentation and safety requirements.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointers,
    /// ignoring the metadata.
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn byte_offset_from<U: ?Sized>(self, origin: NonNullConst<U>) -> isize {
        unsafe { self.0.byte_offset_from(origin.0) }
    }

    // N.B. `wrapping_offset``, `wrapping_add`, etc are not implemented because they can wrap to null

    /// Calculates the distance between two pointers within the same allocation, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of T: the distance in bytes is divided by `size_of::<T>()`.
    ///
    /// This computes the same value that [`offset_from`](#method.offset_from)
    /// would compute, but with the added precondition that the offset is
    /// guaranteed to be non-negative.  This method is equivalent to
    /// `usize::try_from(self.offset_from(origin)).unwrap_unchecked()`,
    /// but it provides slightly more information to the optimizer, which can
    /// sometimes allow it to optimize slightly better with some backends.
    ///
    /// This method can be though of as recovering the `count` that was passed
    /// to [`add`](#method.add) (or, with the parameters in the other order,
    /// to [`sub`](#method.sub)).  The following are all equivalent, assuming
    /// that their safety preconditions are met:
    /// ```rust
    /// # unsafe fn blah(ptr: non_null_const::NonNullConst<u32>, origin: non_null_const::NonNullConst<u32>, count: usize) -> bool { unsafe {
    /// ptr.offset_from_unsigned(origin) == count
    /// # &&
    /// origin.add(count) == ptr
    /// # &&
    /// ptr.sub(count) == origin
    /// # } }
    /// ```
    ///
    /// # Safety
    ///
    /// - The distance between the pointers must be non-negative (`self >= origin`)
    ///
    /// - *All* the safety conditions of [`offset_from`](#method.offset_from)
    ///   apply to this method as well; see it for the full details.
    ///
    /// Importantly, despite the return type of this method being able to represent
    /// a larger offset, it's still *not permitted* to pass pointers which differ
    /// by more than `isize::MAX` *bytes*.  As such, the result of this method will
    /// always be less than or equal to `isize::MAX as usize`.
    ///
    /// # Panics
    ///
    /// This function panics if `T` is a Zero-Sized Type ("ZST").
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// let a = [0; 5];
    /// let ptr1: NonNullConst<u32> = NonNullConst::from(&a[1]);
    /// let ptr2: NonNullConst<u32> = NonNullConst::from(&a[3]);
    /// unsafe {
    ///     assert_eq!(ptr2.offset_from_unsigned(ptr1), 2);
    ///     assert_eq!(ptr1.add(2), ptr2);
    ///     assert_eq!(ptr2.sub(2), ptr1);
    ///     assert_eq!(ptr2.offset_from_unsigned(ptr2), 0);
    /// }
    ///
    /// // This would be incorrect, as the pointers are not correctly ordered:
    /// // ptr1.offset_from_unsigned(ptr2)
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub const unsafe fn offset_from_unsigned(self, subtracted: NonNullConst<T>) -> usize
    where
        T: Sized,
    {
        unsafe { self.0.offset_from_unsigned(subtracted.0) }
    }

    /// Calculates the distance between two pointers within the same allocation, *where it's known that
    /// `self` is equal to or greater than `origin`*. The returned value is in
    /// units of **bytes**.
    ///
    /// This is purely a convenience for casting to a `u8` pointer and
    /// using [`offset_from_unsigned`][NonNullConst::offset_from_unsigned] on it.
    /// See that method for documentation and safety requirements.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointers,
    /// ignoring the metadata.
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn byte_offset_from_unsigned<U: ?Sized>(
        self,
        origin: NonNullConst<U>,
    ) -> usize {
        unsafe { self.0.byte_offset_from_unsigned(origin.0) }
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// See [`ptr::read`] for safety concerns and examples.
    ///
    /// [`ptr::read`]: core::ptr::read()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn read(self) -> T
    where
        T: Sized,
    {
        unsafe { self.0.read() }
    }

    /// Performs a volatile read of the value from `self` without moving it. This
    /// leaves the memory in `self` unchanged.
    ///
    /// Volatile operations are intended to act on I/O memory, and are guaranteed
    /// to not be elided or reordered by the compiler across other volatile
    /// operations.
    ///
    /// See [`ptr::read_volatile`] for safety concerns and examples.
    ///
    /// [`ptr::read_volatile`]: core::ptr::read_volatile()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn read_volatile(self) -> T
    where
        T: Sized,
    {
        unsafe { self.0.read_volatile() }
    }

    /// Reads the value from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// Unlike `read`, the pointer may be unaligned.
    ///
    /// See [`ptr::read_unaligned`] for safety concerns and examples.
    ///
    /// [`ptr::read_unaligned`]: core::ptr::read_unaligned()
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn read_unaligned(self) -> T
    where
        T: Sized,
    {
        unsafe { self.0.read_unaligned() }
    }

    /// Copies `count * size_of::<T>()` bytes from `self` to `dest`. The source
    /// and destination may overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy`].
    ///
    /// See [`ptr::copy`] for safety concerns and examples.
    ///
    /// [`ptr::copy`]: core::ptr::copy()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn copy_to(self, dest: NonNullConst<T>, count: usize)
    where
        T: Sized,
    {
        unsafe {
            self.0.copy_to(dest.0, count);
        }
    }

    /// Copies `count * size_of::<T>()` bytes from `self` to `dest`. The source
    /// and destination may *not* overlap.
    ///
    /// NOTE: this has the *same* argument order as [`ptr::copy_nonoverlapping`].
    ///
    /// See [`ptr::copy_nonoverlapping`] for safety concerns and examples.
    ///
    /// [`ptr::copy_nonoverlapping`]: core::ptr::copy_nonoverlapping()
    #[inline(always)]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn copy_to_nonoverlapping(self, dest: NonNullConst<T>, count: usize)
    where
        T: Sized,
    {
        unsafe {
            self.0.copy_to_nonoverlapping(dest.0, count);
        }
    }

    /// Computes the offset that needs to be applied to the pointer in order to make it aligned to
    /// `align`.
    ///
    /// If it is not possible to align the pointer, the implementation returns
    /// `usize::MAX`.
    ///
    /// The offset is expressed in number of `T` elements, and not bytes.
    ///
    /// There are no guarantees whatsoever that offsetting the pointer will not overflow or go
    /// beyond the allocation that the pointer points into. It is up to the caller to ensure that
    /// the returned offset is correct in all terms other than alignment.
    ///
    /// When this is called during compile-time evaluation (which is unstable), the implementation
    /// may return `usize::MAX` in cases where that can never happen at runtime. This is because the
    /// actual alignment of pointers is not known yet during compile-time, so an offset with
    /// guaranteed alignment can sometimes not be computed. For example, a buffer declared as `[u8;
    /// N]` might be allocated at an odd or an even address, but at compile-time this is not yet
    /// known, so the execution has to be correct for either choice. It is therefore impossible to
    /// find an offset that is guaranteed to be 2-aligned. (This behavior is subject to change, as usual
    /// for unstable APIs.)
    ///
    /// # Panics
    ///
    /// The function panics if `align` is not a power-of-two.
    ///
    /// # Examples
    ///
    /// Accessing adjacent `u8` as `u16`
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// # unsafe {
    /// let x = [5_u8, 6, 7, 8, 9];
    /// let ptr = NonNullConst::new(x.as_ptr() as *const u8).unwrap();
    /// let offset = ptr.align_offset(align_of::<u16>());
    ///
    /// if offset < x.len() - 1 {
    ///     let u16_ptr = ptr.add(offset).cast::<u16>();
    ///     assert!(u16_ptr.read() == u16::from_ne_bytes([5, 6]) || u16_ptr.read() == u16::from_ne_bytes([6, 7]));
    /// } else {
    ///     // while the pointer can be aligned via `offset`, it would point
    ///     // outside the allocation
    /// }
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn align_offset(self, align: usize) -> usize
    where
        T: Sized,
    {
        self.0.align_offset(align)
    }

    /// Returns whether the pointer is properly aligned for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use non_null_const::NonNullConst;
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// let data = AlignedI32(42);
    /// let ptr = NonNullConst::<AlignedI32>::from(&data);
    ///
    /// assert!(ptr.is_aligned());
    /// assert!(!NonNullConst::new(ptr.as_ptr().wrapping_byte_add(1)).unwrap().is_aligned());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_aligned(self) -> bool
    where
        T: Sized,
    {
        self.0.is_aligned()
    }

    /// Returns whether the pointer is aligned to `align`.
    ///
    /// For non-`Sized` pointees this operation considers only the data pointer,
    /// ignoring the metadata.
    ///
    /// # Panics
    ///
    /// The function panics if `align` is not a power-of-two (this includes 0).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pointer_is_aligned_to)]
    ///
    /// // On some platforms, the alignment of i32 is less than 4.
    /// #[repr(align(4))]
    /// struct AlignedI32(i32);
    ///
    /// let data = AlignedI32(42);
    /// let ptr = &data as *const AlignedI32;
    ///
    /// assert!(ptr.is_aligned_to(1));
    /// assert!(ptr.is_aligned_to(2));
    /// assert!(ptr.is_aligned_to(4));
    ///
    /// assert!(ptr.wrapping_byte_add(2).is_aligned_to(2));
    /// assert!(!ptr.wrapping_byte_add(2).is_aligned_to(4));
    ///
    /// assert_ne!(ptr.is_aligned_to(8), ptr.wrapping_add(1).is_aligned_to(8));
    /// ```
    #[inline]
    #[must_use]
    #[cfg(feature = "pointer_is_aligned_to")]
    pub fn is_aligned_to(self, align: usize) -> bool {
        self.0.is_aligned_to(align)
    }
}

impl<T> NonNullConst<T> {
    /// Casts from a type to its maybe-uninitialized version.
    #[must_use]
    #[inline(always)]
    #[cfg(feature = "cast_maybe_uninit")]
    pub const fn cast_uninit(self) -> NonNullConst<MaybeUninit<T>> {
        NonNullConst(self.0.cast())
    }
}
impl<T> NonNullConst<MaybeUninit<T>> {
    /// Casts from a maybe-uninitialized type to its initialized version.
    ///
    /// This is always safe, since UB can only occur if the pointer is read
    /// before being initialized.
    #[must_use]
    #[inline(always)]
    #[cfg(feature = "cast_maybe_uninit")]
    pub const fn cast_init(self) -> NonNullConst<T> {
        NonNullConst(self.0.cast())
    }
}

impl<T> NonNullConst<[T]> {
    /// Creates a non-null raw slice from a thin pointer and a length.
    ///
    /// The `len` argument is the number of **elements**, not the number of bytes.
    ///
    /// This function is safe, but dereferencing the return value is unsafe.
    /// See the documentation of [`slice::from_raw_parts`](core::slice::from_raw_parts) for slice safety requirements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use non_null_const::NonNullConst;
    ///
    /// // create a slice pointer when starting out with a pointer to the first element
    /// let x = [5, 6, 7];
    /// let nonnullconst_pointer = NonNullConst::new(x.as_ptr()).unwrap();
    /// let slice = NonNullConst::slice_from_raw_parts(nonnullconst_pointer, 3);
    /// assert_eq!(unsafe { slice.as_ref()[2] }, 7);
    /// ```
    ///
    /// (Note that this example artificially demonstrates a use of this method,
    /// but `let slice = NonNullConst::from(&x[..]);` would be a better way to write code like this.)
    #[must_use]
    #[inline]
    pub const fn slice_from_raw_parts(data: NonNullConst<T>, len: usize) -> Self {
        Self(NonNullMut::slice_from_raw_parts(data.0, len))
    }

    /// Returns the length of a non-null raw slice.
    ///
    /// The returned value is the number of **elements**, not the number of bytes.
    ///
    /// This function is safe, even when the non-null raw slice cannot be dereferenced to a slice
    /// because the pointer does not have a valid address.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use non_null_const::NonNullConst;
    ///
    /// let slice: NonNullConst<[i8]> = NonNullConst::slice_from_raw_parts(NonNullConst::dangling(), 3);
    /// assert_eq!(slice.len(), 3);
    /// ```
    #[must_use]
    #[inline]
    pub const fn len(self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the non-null raw slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use non_null_const::NonNullConst;
    ///
    /// let slice: NonNullConst<[i8]> = NonNullConst::slice_from_raw_parts(NonNullConst::dangling(), 3);
    /// assert!(!slice.is_empty());
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    /// Returns a non-null pointer to the slice's buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(slice_ptr_get)]
    /// use non_null_const::NonNullConst;
    ///
    /// let slice: NonNullConst<[i8]> = NonNullConst::slice_from_raw_parts(NonNullConst::dangling(), 3);
    /// assert_eq!(slice.as_non_null_ptr(), NonNullConst::<i8>::dangling());
    /// ```
    #[inline]
    #[must_use]
    #[cfg(feature = "slice_ptr_get")]
    pub const fn as_non_null_ptr(self) -> NonNullConst<T> {
        NonNullConst(self.0.as_non_null_ptr())
    }

    /// Returns a shared reference to a slice of possibly uninitialized values. In contrast to
    /// [`as_ref`], this does not require that the value has to be initialized.
    ///
    /// [`as_ref`]: NonNullConst::as_ref
    ///
    /// # Safety
    ///
    /// When calling this method, you have to ensure that all of the following is true:
    ///
    /// * The pointer must be [valid] for reads for `ptr.len() * size_of::<T>()` many bytes,
    ///   and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single allocation!
    ///       Slices can never span across multiple allocations.
    ///
    ///     * The pointer must be aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`NonNullConst::dangling()`].
    ///
    /// * The total size `ptr.len() * size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
    ///   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
    ///
    /// This applies even if the result of this method is unused!
    ///
    /// See also [`slice::from_raw_parts`](core::slice::from_raw_parts).
    ///
    /// [valid]: core::ptr#safety
    #[inline]
    #[must_use]
    #[cfg(feature = "ptr_as_uninit")]
    pub const unsafe fn as_uninit_slice<'a>(self) -> &'a [MaybeUninit<T>] {
        unsafe { self.0.as_uninit_slice() }
    }

    #[cfg(feature = "slice_ptr_get")]
    cfg_tt! {
    /// Returns a raw pointer to an element or subslice, without doing bounds
    /// checking.
    ///
    /// Calling this method with an out-of-bounds index or when `self` is not dereferenceable
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(slice_ptr_get)]
    /// use non_null_const::NonNullConst;
    ///
    /// let x = &[1, 2, 4];
    /// let x = NonNullConst::slice_from_raw_parts(NonNullConst::new(x.as_ptr()).unwrap(), x.len());
    ///
    /// unsafe {
    ///     assert_eq!(x.get_unchecked(1).as_ptr(), x.as_non_null_ptr().as_ptr().add(1));
    /// }
    /// ```
    #[inline]
    pub #[cfg(feature = "const_index")] const unsafe fn get_unchecked<I>(self, index: I) -> NonNullConst<I::Output>
    where
        I: #[cfg(feature = "const_index")]([const]) SliceIndex<[T]>,
    {
        unsafe { NonNullConst(self.0.get_unchecked_mut(index)) }
    }
    }
}

#[allow(clippy::non_canonical_clone_impl)]
impl<T: PointeeSized> Clone for NonNullConst<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T: PointeeSized> Copy for NonNullConst<T> {}

#[cfg(feature = "coerce_unsized")]
impl<T: PointeeSized, U: PointeeSized> CoerceUnsized<NonNullConst<U>> for NonNullConst<T> where
    T: Unsize<U>
{
}

#[cfg(feature = "dispatch_from_dyn")]
impl<T: PointeeSized, U: PointeeSized> DispatchFromDyn<NonNullConst<U>> for NonNullConst<T> where
    T: Unsize<U>
{
}

unsafe impl<T: PointeeSized> PinCoerceUnsized for NonNullConst<T> {}

impl<T: PointeeSized> fmt::Debug for NonNullConst<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: PointeeSized> fmt::Pointer for NonNullConst<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: PointeeSized> Eq for NonNullConst<T> {}

impl<T: PointeeSized> PartialEq for NonNullConst<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: PointeeSized> Ord for NonNullConst<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<T: PointeeSized> PartialOrd for NonNullConst<T> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: PointeeSized> hash::Hash for NonNullConst<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[cfg(feature = "ptr_internals")]
cfg_tt! {
impl<T: PointeeSized> #[cfg(feature = "const_convert")] const From<Unique<T>> for NonNullConst<T> {
    #[inline]
    fn from(unique: Unique<T>) -> Self {
        Self(NonNullMut::from(unique))
    }
}
}

impl<T: PointeeSized> const From<&mut T> for NonNullConst<T> {
    /// Converts a `&mut T` to a `NonNullConst<T>`.
    ///
    /// This conversion is safe and infallible since references cannot be null.
    #[inline]
    fn from(r: &mut T) -> Self {
        Self(NonNullMut::from(r))
    }
}

impl<T: PointeeSized> const From<&T> for NonNullConst<T> {
    /// Converts a `&T` to a `NonNullConst<T>`.
    ///
    /// This conversion is safe and infallible since references cannot be null.
    #[inline]
    fn from(r: &T) -> Self {
        Self(NonNullMut::from(r))
    }
}

// --- Extra traits and methods ---
impl<T: PointeeSized> const From<NonNullMut<T>> for NonNullConst<T> {
    fn from(ptr: NonNullMut<T>) -> Self {
        Self(ptr)
    }
}

impl<T: PointeeSized> const From<NonNullConst<T>> for NonNullMut<T> {
    fn from(ptr: NonNullConst<T>) -> Self {
        ptr.0
    }
}
