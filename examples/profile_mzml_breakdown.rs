use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use memchr::memmem;
use memmap2::Mmap;
use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};
use std::fs::File;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: profile_mzml_breakdown <input .mzML>");
        std::process::exit(2)
    });

    println!("=========================================================================");
    println!("                   mzML Profiling & Breakdown Benchmark                  ");
    println!("=========================================================================\n");
    println!("Target File: {}\n", path);

    // 1. Memory Map Timing
    let t0 = Instant::now();
    let file = File::open(&path).expect("Failed to open file");
    let file_len = file.metadata().unwrap().len();
    let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap") };
    let mmap_time = t0.elapsed();
    println!("1. File Size: {:.2} MB", file_len as f64 / 1_048_576.0);
    println!("   mmap open time: {:?}", mmap_time);

    // 2. Index / Span Discovery Timing
    let t1 = Instant::now();
    let marker = b"<indexListOffset>";
    let marker_pos = memmem::rfind(&mmap, marker);
    let mut is_indexed = false;
    if let Some(_) = marker_pos {
        is_indexed = true;
    }
    // Parse using MzmlParser's internal span logic
    let mut parser = MzmlParser::new(MzmlParserConfig {
        parse_metadata: true,
        use_parallel_parsing: true,
        use_fast_tokenizer: true,
    });
    // We can call parser on file
    let t_parse = Instant::now();
    let records = parser.parse_file(&path).unwrap();
    let total_parse_time = t_parse.elapsed();

    let ms1_records: Vec<_> = records.iter().filter(|r| r.ms_level == 1).collect();
    let ms2_records: Vec<_> = records.iter().filter(|r| r.ms_level >= 2).collect();

    println!("2. Index Status: indexedmzML = {}", is_indexed);
    println!("   Total Spectra: {} ({} MS1, {} MS2)", records.len(), ms1_records.len(), ms2_records.len());
    println!("   Full Parse Time (parallel): {:?}", total_parse_time);
    println!("   Overall Parsing Throughput: {:.1} MB/s | {:.0} spectra/s\n",
        (file_len as f64 / 1_048_576.0) / total_parse_time.as_secs_f64(),
        records.len() as f64 / total_parse_time.as_secs_f64()
    );

    // 3. Detailed Component Micro-benchmarks on 5,000 spectra
    println!("3. Component Micro-benchmarks (isolated sequential timing):");
    // Sample slices from the mmap
    let marker_spectrum = b"<spectrum ";
    let mut offsets = Vec::new();
    let mut cur = 0;
    while let Some(rel) = memmem::find(&mmap[cur..], marker_spectrum) {
        let pos = cur + rel;
        offsets.push(pos);
        cur = pos + marker_spectrum.len();
        if offsets.len() >= 5000 {
            break;
        }
    }

    let sample_count = offsets.len().saturating_sub(1).min(3000);
    println!("   Profiling {} consecutive spectrum blocks sequentially...", sample_count);

    let mut t_xml_scan = std::time::Duration::ZERO;
    let mut t_base64 = std::time::Duration::ZERO;
    let mut t_zlib = std::time::Duration::ZERO;
    let mut t_peak_alloc = std::time::Duration::ZERO;

    let mut decompressor = libdeflater::Decompressor::new();
    let mut decomp_buf = vec![0u8; 1024 * 1024];

    for i in 0..sample_count {
        let start = offsets[i];
        let end = offsets[i + 1];
        let slice = &mmap[start..end];

        // A. XML Scan: finding <binary> boundaries
        let t_a = Instant::now();
        let mut binary_slices = Vec::new();
        let mut pos = 0;
        while let Some(rel_b) = memmem::find(&slice[pos..], b"<binary>") {
            let b_start = pos + rel_b + b"<binary>".len();
            if let Some(rel_end) = memmem::find(&slice[b_start..], b"</binary>") {
                let b_end = b_start + rel_end;
                binary_slices.push(&slice[b_start..b_end]);
                pos = b_end + b"</binary>".len();
            } else {
                break;
            }
        }
        t_xml_scan += t_a.elapsed();

        // B. Base64 Decode
        for b_slice in &binary_slices {
            let trimmed = b_slice.trim_ascii();
            let t_b = Instant::now();
            let raw_res = BASE64.decode(trimmed);
            t_base64 += t_b.elapsed();

            // C. Zlib Decompress
            if let Ok(raw) = raw_res {
                let t_c = Instant::now();
                let decomp_res = decompressor.zlib_decompress(&raw, &mut decomp_buf);
                t_zlib += t_c.elapsed();

                // D. Float conversion
                if let Ok(n_bytes) = decomp_res {
                    let t_d = Instant::now();
                    let n_floats = n_bytes / 8;
                    let mut floats = Vec::with_capacity(n_floats);
                    for chunk in decomp_buf[..n_bytes].chunks_exact(8) {
                        floats.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    t_peak_alloc += t_d.elapsed();
                }
            }
        }
    }

    let sample_total = t_xml_scan + t_base64 + t_zlib + t_peak_alloc;
    println!("   -------------------------------------------------------------------");
    println!("   Component                  Time ({} spectra)   % of Processing", sample_count);
    println!("   -------------------------------------------------------------------");
    println!("   XML Tag / Binary Scanning: {:<15?}   {:.1}%", t_xml_scan, (t_xml_scan.as_secs_f64() / sample_total.as_secs_f64()) * 100.0);
    println!("   Base64 ASCII Decoding:     {:<15?}   {:.1}%", t_base64, (t_base64.as_secs_f64() / sample_total.as_secs_f64()) * 100.0);
    println!("   Zlib Decompression:        {:<15?}   {:.1}%", t_zlib, (t_zlib.as_secs_f64() / sample_total.as_secs_f64()) * 100.0);
    println!("   Float Cast & Allocation:   {:<15?}   {:.1}%", t_peak_alloc, (t_peak_alloc.as_secs_f64() / sample_total.as_secs_f64()) * 100.0);
    println!("   -------------------------------------------------------------------");
    println!("   Sample Total:              {:?}", sample_total);
}
