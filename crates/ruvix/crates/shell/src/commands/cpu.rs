//! CPU information command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Execute the cpu command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let cpus = backend.cpu_info();

    if cpus.is_empty() {
        return String::from("No CPU information available.");
    }

    let online_count = cpus.iter().filter(|c| c.online).count();
    let total_count = cpus.len();

    let mut output = String::from("CPU Information (SMP)\n");
    output.push_str("=====================\n");
    output.push_str(&format!("CPUs: {} online / {} total\n\n", online_count, total_count));
    output.push_str("  ID   STATE     FREQ     LOAD   TYPE\n");
    output.push_str("  ---  --------  -------  -----  --------\n");

    for cpu in &cpus {
        let state = if cpu.online { "ONLINE" } else { "OFFLINE" };
        let cpu_type = if cpu.is_primary { "PRIMARY" } else { "SECONDARY" };
        let freq = if cpu.freq_mhz > 0 {
            format!("{} MHz", cpu.freq_mhz)
        } else {
            String::from("N/A")
        };
        let load = if cpu.online {
            format!("{}%", cpu.load_percent)
        } else {
            String::from("-")
        };

        let line = format!(
            "  {:>3}  {:<8}  {:>7}  {:>5}  {}\n",
            cpu.id, state, freq, load, cpu_type
        );
        output.push_str(&line);
    }

    // Calculate average load
    let online_cpus: alloc::vec::Vec<_> = cpus.iter().filter(|c| c.online).collect();
    if !online_cpus.is_empty() {
        let total_load: u32 = online_cpus.iter().map(|c| u32::from(c.load_percent)).sum();
        let avg_load = total_load / online_cpus.len() as u32;
        output.push_str(&format!("\nAverage load: {}%", avg_load));
    }

    output
}

#[cfg(test)]
mod tests {
    use crate::CpuInfo;

    #[test]
    fn test_cpu_info() {
        let cpu = CpuInfo {
            id: 0,
            online: true,
            is_primary: true,
            freq_mhz: 1800,
            load_percent: 25,
        };
        assert_eq!(cpu.id, 0);
        assert!(cpu.online);
        assert!(cpu.is_primary);
    }
}
