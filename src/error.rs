use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Input is malformed or truncated.
    Invalid(String),
    /// Input is valid but uses a feature we do not handle.
    Unsupported(String),
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(m) => write!(f, "invalid data: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
