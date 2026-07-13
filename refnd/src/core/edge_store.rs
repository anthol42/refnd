use std::{
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
use super::leiden::{CsrGraph, INWeightType};

// ── Public struct ─────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct EdgeStore {
    pub node_count: usize,
    edges: Vec<(u32, u32, f32)>,
}

impl EdgeStore {
    /// Build from a list of `(u, v, weight)` triples.
    pub fn new(node_count: usize, edges: Vec<(u32, u32, f32)>) -> Self {
        Self { node_count, edges }
    }

    /// Return a slice of all edges as `(src, dst, weight)` triples.
    pub fn edges(&self) -> &[(u32, u32, f32)] {
        &self.edges
    }

    /// Build a [`CsrGraph`] from the stored edges.
    pub fn graph(&self, inweight_type: INWeightType) -> CsrGraph {
        CsrGraph::new(self.node_count, &self.edges, inweight_type)
    }

    /// Persist to `path`.
    ///
    /// - `.edgelist` → UTF-8 text: `# n = <count>` header, then `u v w` per line.
    /// - `.edgestr`  → compact binary (see [`Self::load`] for layout).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let path = path.as_ref();
        match extension(path)? {
            "edgelist" => self.save_text(path),
            "edgestr"  => self.save_binary(path),
            ext => Err(format!("unknown extension '.{ext}': expected .edgelist or .edgestr").into()),
        }
    }

    /// Load from `path`. Extension determines the format (see [`Self::save`]).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        match extension(path)? {
            "edgelist" => Self::load_text(path),
            "edgestr"  => Self::load_binary(path),
            ext => Err(format!("unknown extension '.{ext}': expected .edgelist or .edgestr").into()),
        }
    }

    // ── Text ──────────────────────────────────────────────────────────────────

    pub fn save_text(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "# n = {}", self.node_count)?;
        for &(u, v, weight) in &self.edges {
            writeln!(w, "{u} {v} {weight}")?;
        }
        Ok(())
    }

    pub fn load_text(path: &Path) -> Result<Self, Box<dyn Error>> {
        let reader = BufReader::new(File::open(path)?);
        let mut lines = reader.lines();

        // Header: "# n = <count>"
        let header = lines.next().ok_or("file is empty")??;
        let node_count: usize = header
            .strip_prefix("# n = ")
            .ok_or_else(|| format!("malformed header: {header:?}"))?
            .trim()
            .parse()?;

        let mut edges: Vec<(u32, u32, f32)> = Vec::new();
        for (i, line) in lines.enumerate() {
            let line = line?;
            let mut parts = line.split_ascii_whitespace();
            let u: u32 = parts.next().ok_or_else(|| format!("line {}: missing u", i + 2))?.parse()?;
            let v: u32 = parts.next().ok_or_else(|| format!("line {}: missing v", i + 2))?.parse()?;
            let w: f32 = parts.next().ok_or_else(|| format!("line {}: missing w", i + 2))?.parse()?;
            edges.push((u, v, w));
        }

        Ok(Self::new(node_count, edges))
    }

    // ── Binary ────────────────────────────────────────────────────────────────
    //
    // Layout:
    //   [6 bytes] crate_version (major, minor, patch), each u16 LE
    //   [8 bytes] node_count  as u64 LE
    //   [8 bytes] edge_count  as u64 LE
    //   per edge: u32 + u32 + f32 = 12 bytes, all little-endian
    //
    // The crate is pre-1.0; the format is only guaranteed stable from v0.1.0
    // onward (see `crate::core::hnsw::current_crate_version`).
    // `load_binary` rejects files saved before that.

    pub fn save_binary(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut w = BufWriter::new(File::create(path)?);
        let (major, minor, patch) = crate::core::hnsw::current_crate_version();
        w.write_all(&major.to_le_bytes())?;
        w.write_all(&minor.to_le_bytes())?;
        w.write_all(&patch.to_le_bytes())?;
        w.write_all(&(self.node_count as u64).to_le_bytes())?;
        w.write_all(&(self.edges.len() as u64).to_le_bytes())?;
        for &(u, v, weight) in &self.edges {
            w.write_all(&u.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            w.write_all(&weight.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load_binary(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut r = BufReader::new(File::open(path)?);

        let version = (read_u16(&mut r)?, read_u16(&mut r)?, read_u16(&mut r)?);
        let running_version = crate::core::hnsw::current_crate_version();
        if version != running_version {
            return Err(format!(
                "edgestr format mismatch: saved with refnd v{}.{}.{}, running v{}.{}.{} — \
                 one of them predates the stable edgestr format. Regenerate the file.",
                version.0, version.1, version.2,
                running_version.0, running_version.1, running_version.2,
            ).into());
        }

        let node_count = read_u64(&mut r)? as usize;
        let edge_count = read_u64(&mut r)? as usize;

        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let u = read_u32(&mut r)?;
            let v = read_u32(&mut r)?;
            let w = read_f32(&mut r)?;
            edges.push((u, v, w));
        }
        Ok(Self { node_count, edges })
    }
}

// ── Index / Iterator ──────────────────────────────────────────────────────────

impl EdgeStore {
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn get(&self, idx: usize) -> (u32, u32, f32) {
        self.edges[idx]
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, f32)> + '_ {
        self.edges.iter().copied()
    }
}

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Display for EdgeStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeStore(n={})", self.len())
    }
}

impl fmt::Debug for EdgeStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.len();
        const MAX: usize = 10;
        write!(f, "[")?;
        if n <= MAX {
            for (i, &(u, v, w)) in self.edges.iter().enumerate() {
                if i > 0 { write!(f, "\n ")?; }
                write!(f, " {u}, {v}, {w},")?;
            }
        } else {
            let head = MAX / 2;
            let tail = MAX - head;
            for (i, &(u, v, w)) in self.edges[..head].iter().enumerate() {
                if i > 0 { write!(f, "\n ")?; }
                write!(f, " {u}, {v}, {w},")?;
            }
            write!(f, "\n   ...")?;
            for &(u, v, w) in &self.edges[n - tail..] {
                write!(f, "\n  {u}, {v}, {w},")?;
            }
        }
        write!(f, "\n]")
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extension<'a>(path: &'a Path) -> Result<&'a str, Box<dyn Error>> {
    path.extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| format!("cannot determine extension of '{}'", path.display()).into())
}

fn read_u16(r: &mut impl std::io::Read) -> Result<u16, Box<dyn Error>> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl std::io::Read) -> Result<u32, Box<dyn Error>> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl std::io::Read) -> Result<u64, Box<dyn Error>> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl std::io::Read) -> Result<f32, Box<dyn Error>> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
