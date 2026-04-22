//! Command line parser for the RuVix debug shell.
//!
//! This module implements a simple line-based parser that handles
//! command parsing and argument extraction for the debug shell.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Parsed command representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Show help information.
    Help,
    /// Show kernel information.
    Info,
    /// Show memory statistics.
    Mem,
    /// Show task list.
    Tasks,
    /// Show capability table.
    Caps {
        /// Optional task ID to filter by.
        task_id: Option<u32>,
    },
    /// Show queue statistics.
    Queues,
    /// Show vector store info.
    Vectors,
    /// Show proof statistics.
    Proofs,
    /// Show CPU information.
    Cpu,
    /// Show witness log entries.
    Witness {
        /// Number of entries to show (default: 10).
        count: usize,
    },
    /// Show performance counters.
    Perf,
    /// Toggle or show syscall tracing.
    Trace {
        /// None = show status, Some(true) = enable, Some(false) = disable.
        enable: Option<bool>,
    },
    /// Trigger system reboot.
    Reboot,
}

/// Parse error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Unknown command.
    UnknownCommand(String),
    /// Invalid argument.
    InvalidArgument {
        /// The command that received the invalid argument.
        command: String,
        /// The invalid argument value.
        argument: String,
        /// Expected format description.
        expected: String,
    },
    /// Empty input.
    EmptyInput,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCommand(cmd) => write!(f, "Unknown command: '{}'", cmd),
            Self::InvalidArgument {
                command,
                argument,
                expected,
            } => write!(
                f,
                "Invalid argument '{}' for command '{}': expected {}",
                argument, command, expected
            ),
            Self::EmptyInput => write!(f, "Empty input"),
        }
    }
}

/// Command line parser.
#[derive(Debug, Default)]
pub struct Parser {
    _private: (),
}

impl Parser {
    /// Create a new parser.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Parse a command line into a Command.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` if the command is unknown or arguments are invalid.
    pub fn parse(&self, line: &str) -> Result<Command, ParseError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(ParseError::EmptyInput);
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let cmd = tokens[0].to_lowercase();
        let args = &tokens[1..];

        match cmd.as_str() {
            "help" | "?" | "h" => Ok(Command::Help),
            "info" | "version" => Ok(Command::Info),
            "mem" | "memory" => Ok(Command::Mem),
            "tasks" | "ps" | "processes" => Ok(Command::Tasks),
            "caps" | "capabilities" => self.parse_caps(args),
            "queues" | "queue" | "q" => Ok(Command::Queues),
            "vectors" | "vec" | "v" => Ok(Command::Vectors),
            "proofs" | "proof" | "p" => Ok(Command::Proofs),
            "cpu" | "cpus" | "smp" => Ok(Command::Cpu),
            "witness" | "wit" | "w" => self.parse_witness(args),
            "perf" | "performance" | "counters" => Ok(Command::Perf),
            "trace" | "strace" => self.parse_trace(args),
            "reboot" | "restart" | "reset" => Ok(Command::Reboot),
            _ => Err(ParseError::UnknownCommand(cmd)),
        }
    }

    /// Parse the `caps` command arguments.
    fn parse_caps(&self, args: &[&str]) -> Result<Command, ParseError> {
        let task_id = if args.is_empty() {
            None
        } else {
            match args[0].parse::<u32>() {
                Ok(id) => Some(id),
                Err(_) => {
                    return Err(ParseError::InvalidArgument {
                        command: "caps".to_string(),
                        argument: args[0].to_string(),
                        expected: "task ID (u32)".to_string(),
                    })
                }
            }
        };
        Ok(Command::Caps { task_id })
    }

    /// Parse the `witness` command arguments.
    fn parse_witness(&self, args: &[&str]) -> Result<Command, ParseError> {
        let count = if args.is_empty() {
            10 // Default to 10 entries
        } else {
            match args[0].parse::<usize>() {
                Ok(n) => n,
                Err(_) => {
                    return Err(ParseError::InvalidArgument {
                        command: "witness".to_string(),
                        argument: args[0].to_string(),
                        expected: "entry count (usize)".to_string(),
                    })
                }
            }
        };
        Ok(Command::Witness { count })
    }

    /// Parse the `trace` command arguments.
    fn parse_trace(&self, args: &[&str]) -> Result<Command, ParseError> {
        let enable = if args.is_empty() {
            None
        } else {
            match args[0].to_lowercase().as_str() {
                "on" | "enable" | "1" | "true" | "yes" => Some(true),
                "off" | "disable" | "0" | "false" | "no" => Some(false),
                _ => {
                    return Err(ParseError::InvalidArgument {
                        command: "trace".to_string(),
                        argument: args[0].to_string(),
                        expected: "on/off, enable/disable, 1/0, true/false, or yes/no".to_string(),
                    })
                }
            }
        };
        Ok(Command::Trace { enable })
    }

    /// Get all available command names (for completion).
    #[must_use]
    pub fn command_names(&self) -> &'static [&'static str] {
        &[
            "help", "info", "mem", "tasks", "caps", "queues", "vectors", "proofs", "cpu",
            "witness", "perf", "trace", "reboot",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help() {
        let parser = Parser::new();
        assert_eq!(parser.parse("help"), Ok(Command::Help));
        assert_eq!(parser.parse("?"), Ok(Command::Help));
        assert_eq!(parser.parse("h"), Ok(Command::Help));
        assert_eq!(parser.parse("HELP"), Ok(Command::Help));
    }

    #[test]
    fn test_parse_info() {
        let parser = Parser::new();
        assert_eq!(parser.parse("info"), Ok(Command::Info));
        assert_eq!(parser.parse("version"), Ok(Command::Info));
    }

    #[test]
    fn test_parse_mem() {
        let parser = Parser::new();
        assert_eq!(parser.parse("mem"), Ok(Command::Mem));
        assert_eq!(parser.parse("memory"), Ok(Command::Mem));
    }

    #[test]
    fn test_parse_tasks() {
        let parser = Parser::new();
        assert_eq!(parser.parse("tasks"), Ok(Command::Tasks));
        assert_eq!(parser.parse("ps"), Ok(Command::Tasks));
        assert_eq!(parser.parse("processes"), Ok(Command::Tasks));
    }

    #[test]
    fn test_parse_caps() {
        let parser = Parser::new();
        assert_eq!(parser.parse("caps"), Ok(Command::Caps { task_id: None }));
        assert_eq!(
            parser.parse("caps 42"),
            Ok(Command::Caps { task_id: Some(42) })
        );
        assert!(matches!(
            parser.parse("caps invalid"),
            Err(ParseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_parse_queues() {
        let parser = Parser::new();
        assert_eq!(parser.parse("queues"), Ok(Command::Queues));
        assert_eq!(parser.parse("queue"), Ok(Command::Queues));
        assert_eq!(parser.parse("q"), Ok(Command::Queues));
    }

    #[test]
    fn test_parse_vectors() {
        let parser = Parser::new();
        assert_eq!(parser.parse("vectors"), Ok(Command::Vectors));
        assert_eq!(parser.parse("vec"), Ok(Command::Vectors));
        assert_eq!(parser.parse("v"), Ok(Command::Vectors));
    }

    #[test]
    fn test_parse_proofs() {
        let parser = Parser::new();
        assert_eq!(parser.parse("proofs"), Ok(Command::Proofs));
        assert_eq!(parser.parse("proof"), Ok(Command::Proofs));
        assert_eq!(parser.parse("p"), Ok(Command::Proofs));
    }

    #[test]
    fn test_parse_cpu() {
        let parser = Parser::new();
        assert_eq!(parser.parse("cpu"), Ok(Command::Cpu));
        assert_eq!(parser.parse("cpus"), Ok(Command::Cpu));
        assert_eq!(parser.parse("smp"), Ok(Command::Cpu));
    }

    #[test]
    fn test_parse_witness() {
        let parser = Parser::new();
        assert_eq!(parser.parse("witness"), Ok(Command::Witness { count: 10 }));
        assert_eq!(parser.parse("witness 5"), Ok(Command::Witness { count: 5 }));
        assert_eq!(parser.parse("wit 20"), Ok(Command::Witness { count: 20 }));
        assert!(matches!(
            parser.parse("witness invalid"),
            Err(ParseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_parse_perf() {
        let parser = Parser::new();
        assert_eq!(parser.parse("perf"), Ok(Command::Perf));
        assert_eq!(parser.parse("performance"), Ok(Command::Perf));
        assert_eq!(parser.parse("counters"), Ok(Command::Perf));
    }

    #[test]
    fn test_parse_trace() {
        let parser = Parser::new();
        assert_eq!(parser.parse("trace"), Ok(Command::Trace { enable: None }));
        assert_eq!(
            parser.parse("trace on"),
            Ok(Command::Trace { enable: Some(true) })
        );
        assert_eq!(
            parser.parse("trace off"),
            Ok(Command::Trace {
                enable: Some(false)
            })
        );
        assert_eq!(
            parser.parse("trace enable"),
            Ok(Command::Trace { enable: Some(true) })
        );
        assert_eq!(
            parser.parse("trace disable"),
            Ok(Command::Trace {
                enable: Some(false)
            })
        );
        assert_eq!(
            parser.parse("trace 1"),
            Ok(Command::Trace { enable: Some(true) })
        );
        assert_eq!(
            parser.parse("trace 0"),
            Ok(Command::Trace {
                enable: Some(false)
            })
        );
        assert!(matches!(
            parser.parse("trace invalid"),
            Err(ParseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_parse_reboot() {
        let parser = Parser::new();
        assert_eq!(parser.parse("reboot"), Ok(Command::Reboot));
        assert_eq!(parser.parse("restart"), Ok(Command::Reboot));
        assert_eq!(parser.parse("reset"), Ok(Command::Reboot));
    }

    #[test]
    fn test_parse_unknown() {
        let parser = Parser::new();
        assert!(matches!(
            parser.parse("unknown"),
            Err(ParseError::UnknownCommand(_))
        ));
    }

    #[test]
    fn test_parse_empty() {
        let parser = Parser::new();
        assert_eq!(parser.parse(""), Err(ParseError::EmptyInput));
        assert_eq!(parser.parse("   "), Err(ParseError::EmptyInput));
    }

    #[test]
    fn test_parse_whitespace_handling() {
        let parser = Parser::new();
        assert_eq!(parser.parse("  help  "), Ok(Command::Help));
        assert_eq!(
            parser.parse("  caps   42  "),
            Ok(Command::Caps { task_id: Some(42) })
        );
    }

    #[test]
    fn test_command_names() {
        let parser = Parser::new();
        let names = parser.command_names();
        assert!(names.contains(&"help"));
        assert!(names.contains(&"info"));
        assert!(names.contains(&"reboot"));
        assert_eq!(names.len(), 13);
    }
}
