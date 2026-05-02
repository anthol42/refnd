pub trait Distance<T: ?Sized>: Sync
where
    T: Sync
{
    fn call(&self, ref_sample: &T, query: &T) -> f32;
}