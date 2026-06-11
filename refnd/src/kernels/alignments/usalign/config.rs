/// Which length to use when normalizing the TM-score.
///
/// USalign returns two scores: TM1 (normalized by query length) and TM2
/// (normalized by target length). `Min` is the most conservative choice and
/// the default.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum NormMode {
    /// min(TM1, TM2) — conservative, independent of which chain is "query"
    #[default]
    Min,
    /// TM1 — normalized by the length of the first argument to `call()`
    Query,
    /// TM2 — normalized by the length of the second argument to `call()`
    Target,
}
