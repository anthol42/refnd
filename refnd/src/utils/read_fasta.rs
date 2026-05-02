use std::fs;
use std::path::Path;

/// Reads a FASTA file and returns a list of (header, sequence) tuples
/// in the order they appear in the file. The header is returned without
/// the leading '>'. Supports multiline sequences (sequence split across
/// multiple lines).
pub fn read_fasta(path: &Path) -> Result<Vec<(String, String)>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;

    let mut entries: Vec<(String, String)> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            entries.push((header.to_string(), String::new()));
        } else if let Some(last) = entries.last_mut() {
            last.1.push_str(line);
        }
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_fasta(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_single_line_fasta() {
        let input = ">seq1 first protein\nACDEFGH\n>seq2 second protein\nIKLMNPQ\n>seq3\nRSTVWY\n";
        let path = write_temp_fasta("test_single_line.fasta", input);
        let entries = read_fasta(&path).unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], ("seq1 first protein".to_string(), "ACDEFGH".to_string()));
        assert_eq!(entries[1], ("seq2 second protein".to_string(), "IKLMNPQ".to_string()));
        assert_eq!(entries[2], ("seq3".to_string(), "RSTVWY".to_string()));
    }

    #[test]
    fn test_multiline_fasta() {
        let input = ">seq1 split across lines\nACDE\nFGHI\nKLMN\n>seq2 two lines\nPQRS\nTVWY\n";
        let path = write_temp_fasta("test_multiline.fasta", input);
        let entries = read_fasta(&path).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], ("seq1 split across lines".to_string(), "ACDEFGHIKLMN".to_string()));
        assert_eq!(entries[1], ("seq2 two lines".to_string(), "PQRSTVWY".to_string()));
    }
}
