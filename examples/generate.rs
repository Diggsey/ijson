use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_current_dir("test_data")?;
    let mut deque = VecDeque::new();
    for i in 0..100 {
        let mut cmd = Command::new("dummyjson.cmd");

        deque.push_back(
            cmd.arg("template.hbs")
                .stdout(File::create(format!("rnd{:04}.json", i))?)
                .spawn()?,
        );
        if deque.len() >= 20 {
            deque.pop_front().unwrap().wait()?;
        }
    }
    for mut child in deque {
        child.wait()?;
    }
    Ok(())
}
