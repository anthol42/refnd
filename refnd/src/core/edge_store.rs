use std::{
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
use super::leiden::CsrGraph;

// ── Storage ───────────────────────────────────────────────────────────────────
#[derive(Clone)]
enum EdgeStorage {
    U32(Vec<(u32, u32, f32)>),
    U64(Vec<(u64, u64, f32)>),
}

// ── Public struct ─────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct EdgeStore {
    pub node_count: usize,
    edges: EdgeStorage,
}

impl EdgeStore {
    /// Build from a list of `(u, v, weight)` triples.
    /// Automatically uses `U32` storage unless `node_count` exceeds `u32::MAX`.
    pub fn new(node_count: usize, edges: Vec<(usize, usize, f32)>) -> Self {
        let storage = if node_count <= u32::MAX as usize {
            EdgeStorage::U32(
                edges.iter().map(|&(u, v, w)| (u as u32, v as u32, w)).collect(),
            )
        } else {
            EdgeStorage::U64(
                edges.iter().map(|&(u, v, w)| (u as u64, v as u64, w)).collect(),
            )
        };
        Self { node_count, edges: storage }
    }

    /// Return all edges as `(usize, usize, f32)`, regardless of internal storage width.
    pub fn edges(&self) -> Vec<(usize, usize, f32)> {
        match &self.edges {
            EdgeStorage::U32(v) => v.iter().map(|&(u, v, w)| (u as usize, v as usize, w)).collect(),
            EdgeStorage::U64(v) => v.iter().map(|&(u, v, w)| (u as usize, v as usize, w)).collect(),
        }
    }

    /// Build a [`CsrGraph`] from the stored edges. Edges are assumed distances and converted to 
    /// similarities using the 1 / (1+w) formula
    pub fn graph(&self, use_weight: bool, is_weight_distance: bool) -> CsrGraph {
        CsrGraph::new(self.node_count, &self.edges(), use_weight, is_weight_distance)
    }

    /// Persist to `path`.
    ///
    /// - `.edgelist` → UTF-8 text: `# n = <count>` header, then `u v w` per line.
    /// - `.edgestr` → compact binary (see [`Self::load`] for layout).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let path = path.as_ref();
        match extension(path)? {
            "edgelist"  => self.save_text(path),
            "edgestr" => self.save_binary(path),
            ext => Err(format!("unknown extension '.{ext}': expected .edgelist or .edgestr").into()),
        }
    }

    /// Load from `path`. Extension determines the format (see [`Self::save`]).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        match extension(path)? {
            "edgelist"  => Self::load_text(path),
            "edgestr" => Self::load_binary(path),
            ext => Err(format!("unknown extension '.{ext}': expected .edgelist or .edgestr").into()),
        }
    }

    // ── Text ──────────────────────────────────────────────────────────────────

    pub fn save_text(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "# n = {}", self.node_count)?;
        for (u, v, weight) in self.edges() {
            writeln!(w, "{u} {v} {weight}")?;
        }
        Ok(())
    }

    pub fn load_text(path: &Path) -> Result<Self, Box<dyn Error>> {
        let reader = BufReader::new(File::open(path)?);
        let mut lines = reader.lines();

        // Header: "n = <count>"
        let header = lines.next().ok_or("file is empty")??;
        let node_count: usize = header
            .strip_prefix("# n = ")
            .ok_or_else(|| format!("malformed header: {header:?}"))?
            .trim()
            .parse()?;

        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for (i, line) in lines.enumerate() {
            let line = line?;
            let mut parts = line.split_ascii_whitespace();
            let u: usize = parts.next().ok_or_else(|| format!("line {}: missing u", i + 2))?.parse()?;
            let v: usize = parts.next().ok_or_else(|| format!("line {}: missing v", i + 2))?.parse()?;
            let w: f32   = parts.next().ok_or_else(|| format!("line {}: missing w", i + 2))?.parse()?;
            edges.push((u, v, w));
        }

        Ok(Self::new(node_count, edges))
    }

    // ── Binary ────────────────────────────────────────────────────────────────
    //
    // Layout:
    //   [1 byte]  flag — 0x00 = U32 indices, 0x01 = U64 indices
    //   [8 bytes] node_count  as u64 LE
    //   [8 bytes] edge_count  as u64 LE
    //   per edge (U32): 4 + 4 + 4 = 12 bytes
    //   per edge (U64): 8 + 8 + 4 = 20 bytes
    //   all values little-endian

    pub fn save_binary(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut w = BufWriter::new(File::create(path)?);

        match &self.edges {
            EdgeStorage::U32(edges) => {
                w.write_all(&[0x00])?;
                w.write_all(&(self.node_count as u64).to_le_bytes())?;
                w.write_all(&(edges.len() as u64).to_le_bytes())?;
                for &(u, v, weight) in edges {
                    w.write_all(&u.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                    w.write_all(&weight.to_le_bytes())?;
                }
            }
            EdgeStorage::U64(edges) => {
                w.write_all(&[0x01])?;
                w.write_all(&(self.node_count as u64).to_le_bytes())?;
                w.write_all(&(edges.len() as u64).to_le_bytes())?;
                for &(u, v, weight) in edges {
                    w.write_all(&u.to_le_bytes())?;
                    w.write_all(&v.to_le_bytes())?;
                    w.write_all(&weight.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn load_binary(path: &Path) -> Result<Self, Box<dyn Error>> {
        use std::io::Read;
        let mut r = BufReader::new(File::open(path)?);

        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;

        let node_count = read_u64(&mut r)? as usize;
        let edge_count = read_u64(&mut r)? as usize;

        match flag[0] {
            0x00 => {
                let mut edges = Vec::with_capacity(edge_count);
                for _ in 0..edge_count {
                    let u = read_u32(&mut r)?;
                    let v = read_u32(&mut r)?;
                    let w = read_f32(&mut r)?;
                    edges.push((u, v, w));
                }
                Ok(Self { node_count, edges: EdgeStorage::U32(edges) })
            }
            0x01 => {
                let mut edges = Vec::with_capacity(edge_count);
                for _ in 0..edge_count {
                    let u = read_u64(&mut r)?;
                    let v = read_u64(&mut r)?;
                    let w = read_f32(&mut r)?;
                    edges.push((u, v, w));
                }
                Ok(Self { node_count, edges: EdgeStorage::U64(edges) })
            }
            b => Err(format!("unknown store-width flag: 0x{b:02X}").into()),
        }
    }
}

// ── Index / Iterator ──────────────────────────────────────────────────────────

impl EdgeStore {
    pub fn len(&self) -> usize {
        match &self.edges {
            EdgeStorage::U32(v) => v.len(),
            EdgeStorage::U64(v) => v.len(),
        }
    }

    pub fn get(&self, idx: usize) -> (usize, usize, f32) {
        match &self.edges {
            EdgeStorage::U32(v) => { let (u, w, x) = v[idx]; (u as usize, w as usize, x) }
            EdgeStorage::U64(v) => { let (u, w, x) = v[idx]; (u as usize, w as usize, x) }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        (0..self.len()).map(move |i| self.get(i))
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
            for i in 0..n {
                let (u, v, w) = self.get(i);
                if i > 0 { write!(f, "\n ")?; }
                write!(f, " {u}, {v}, {w},")?;
            }
        } else {
            let head = MAX / 2;
            let tail = MAX - head;
            for i in 0..head {
                let (u, v, w) = self.get(i);
                if i > 0 { write!(f, "\n ")?; }
                write!(f, " {u}, {v}, {w},")?;
            }
            write!(f, "\n   ...")?;
            for i in (n - tail)..n {
                let (u, v, w) = self.get(i);
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
