
// Remap community IDs to be contiguous [0, k) so scratch buffers
// indexed by community ID stay in bounds in recursive calls.
// `max_id` is an upper bound on the current community IDs (= g.n of the caller).
// Return the number of unique communities
pub fn reindex_membership(partition: &mut Vec<usize>, max_id: usize) -> usize{
    let mut remap = vec![usize::MAX; max_id];
    let mut next_id = 0usize;
    for v_com_id in partition.iter_mut() {
        let assigned_community = &mut remap[*v_com_id as usize];
        if *assigned_community == usize::MAX {
            *assigned_community = next_id;
            next_id += 1;
        }
        *v_com_id = *assigned_community;
    }
    next_id
}