use std::alloc::System;
use std::error::Error;
use std::fs;

use mockalloc::{AllocInfo, Mockalloc};

#[global_allocator]
static ALLOCATOR: Mockalloc<System> = Mockalloc(System);

fn test_json_decode(data: &[u8]) -> (AllocInfo, AllocInfo) {
    // Measure peak
    let res1 = mockalloc::record_allocs(|| {
        let _value: serde_json::Value = serde_json::from_slice(data).unwrap();
    });
    res1.result().unwrap();

    // Measure clone cost
    let value: serde_json::Value = serde_json::from_slice(data).unwrap();
    let res2 = mockalloc::record_allocs(|| {
        let _value = value.clone();
    });
    res2.result().unwrap();
    (res1, res2)
}

fn test_ijson_decode(data: &[u8]) -> (AllocInfo, AllocInfo) {
    // Measure peak
    let res1 = mockalloc::record_allocs(|| {
        let _value: ijson::IValue = serde_json::from_slice(data).unwrap();
    });
    res1.result().unwrap();

    // Measure clone cost
    let value: ijson::IValue = serde_json::from_slice(data).unwrap();
    let res2 = mockalloc::record_allocs(|| {
        let _value = value.clone();
    });
    res2.result().unwrap();
    (res1, res2)
}

fn print_alloc_info(
    name: &str,
    size: usize,
    alloc_info1: (AllocInfo, AllocInfo),
    alloc_info2: (AllocInfo, AllocInfo),
) {
    println!(
        "{:?},{},{},{},{},{},{},{},{},{}",
        name,
        size,
        alloc_info1.0.peak_mem(),
        alloc_info2.0.peak_mem(),
        alloc_info1.0.num_allocs(),
        alloc_info2.0.num_allocs(),
        alloc_info1.1.peak_mem(),
        alloc_info2.1.peak_mem(),
        alloc_info1.1.num_allocs(),
        alloc_info2.1.num_allocs(),
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    // The string cache is normally lazily initialized which would erroneously show up as a
    // memory leak, so explicitly initialize it here.
    ijson::string::init_cache();
    println!(
        r#""Filename","JSON size (B)","serde-json peak memory usage (B)","ijson peak memory usage (B)","serde-json allocations","ijson allocations","serde-json clone memory usage (B)","ijson clone memory usage (B)","serde-json clone allocations","ijson clone allocations""#
    );
    for test_file in fs::read_dir("test_data")? {
        let test_file = test_file?;
        if !test_file.file_type()?.is_file() {
            continue;
        }
        let path = test_file.path();
        if path.extension() != Some("json".as_ref()) {
            continue;
        }
        let contents = fs::read(test_file.path())?;

        let json_info = test_json_decode(&contents);
        let ijson_info = test_ijson_decode(&contents);
        let name = test_file.file_name().to_string_lossy().to_string();
        print_alloc_info(&name, contents.len(), json_info, ijson_info);
    }
    Ok(())
}
