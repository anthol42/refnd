use parasail_rs::prelude::*;
use super::matrix::BundledMatrix;

pub enum AlignMode {
    Global,
    Local,
    SemiGlobal
}

pub enum DatatypeWidth {
    Short = 8,
    Half = 16,
    Full = 32,
    Long = 64,
    Sat
}

pub enum AlignerMatrix {
    Bundled(BundledMatrix),
    Custom(Matrix)
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum VectorizationStrategy {
    Striped,
    Scan,
    Diag,
}
pub struct AlignerConfig {
    pub mode: AlignMode,
    pub width: DatatypeWidth,
    pub matrix: AlignerMatrix,
    pub vectorization: VectorizationStrategy,
    
    // Should be strictly positive
    pub gap_open: i32,
    pub gap_extend: i32,
}
pub trait AlignerConfigTrait {
    fn aligner_cfg(&mut self) -> &mut AlignerConfig;
    fn aligner_cfg_ref(&self) -> &AlignerConfig;

    fn build_aligner(&self) -> Aligner {
        let cfg = self.aligner_cfg_ref();
        let mut builder = AlignerBuilder::default();

        match cfg.mode {
            AlignMode::Global =>     { builder.global(); }
            AlignMode::Local =>      { builder.local(); }
            AlignMode::SemiGlobal => { builder.semi_global(); }
        }

        match cfg.width {
            DatatypeWidth::Sat =>   {}
            DatatypeWidth::Short => { builder.solution_width(8); }
            DatatypeWidth::Half =>  { builder.solution_width(16); }
            DatatypeWidth::Full =>  { builder.solution_width(32); }
            DatatypeWidth::Long =>  { builder.solution_width(64); }
        }

        match &cfg.matrix {
            AlignerMatrix::Bundled(m) => {
                builder.matrix(Matrix::from(m.to_parasail_name()).expect("Invalid matrix name"));
            }
            AlignerMatrix::Custom(m) => {
                builder.matrix(m.clone());
            }
        }

        match cfg.vectorization {
            VectorizationStrategy::Striped => { builder.striped(); }
            VectorizationStrategy::Scan =>    { builder.scan(); }
            VectorizationStrategy::Diag =>    { builder.diag(); }
        }

        builder.gap_open(cfg.gap_open);
        builder.gap_extend(cfg.gap_extend);
        builder.use_stats();

        builder.build()
    }

    fn set_mode(&mut self, mode: AlignMode) -> &mut Self{
        self.aligner_cfg().mode = mode;
        self
    }

    fn set_width(&mut self, width: DatatypeWidth) -> &mut Self{
        self.aligner_cfg().width = width;
        self
    }

    fn set_matrix(&mut self, matrix: AlignerMatrix) -> &mut Self{
        self.aligner_cfg().matrix = matrix;
        self
    }
    fn set_vectorization(&mut self, strategy: VectorizationStrategy) -> &mut Self{
        self.aligner_cfg().vectorization = strategy;
        self
    }
    fn set_gap_open(&mut self, gap_open: i32) -> &mut Self{
        if gap_open < 0{
            panic!("Gap open must be greater than zero");
        }
        self.aligner_cfg().gap_open = gap_open;
        self
    }
    fn set_gap_extend(&mut self, gap_extend: i32) -> &mut Self{
        if gap_extend < 0{
            panic!("Gap extend must be greater than zero");
        }
        self.aligner_cfg().gap_extend = gap_extend;
        self
    }
}