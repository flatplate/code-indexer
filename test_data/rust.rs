// Embedding test file for Rust

/// Represents a user's profile information
#[derive(Debug, Clone)]
pub struct UserProfile {
    id: u64,
    username: String,
    email: String,
    preferences: std::collections::HashMap<String, String>,
}

/// Defines different user role levels
#[derive(Debug, Clone, PartialEq)]
pub enum UserRole {
    Admin,
    Moderator,
    User,
    Guest,
}

/// Result type for validation operations
#[derive(Debug)]
pub struct ValidationResult<T> {
    is_valid: bool,
    errors: Option<Vec<String>>,
    data: Option<T>,
}

/// Database connection singleton implementation
pub struct DatabaseConnection {
    connection_string: String,
}

impl DatabaseConnection {
    /// Static instance holder
    static mut INSTANCE: Option<DatabaseConnection> = None;

    /// Private constructor
    fn new(connection_string: String) -> Self {
        Self { connection_string }
    }

    /// Get or create singleton instance
    pub fn get_instance(connection_string: String) -> &'static DatabaseConnection {
        unsafe {
            if INSTANCE.is_none() {
                INSTANCE = Some(DatabaseConnection::new(connection_string));
            }
            INSTANCE.as_ref().unwrap()
        }
    }

    pub async fn connect(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Simulated connection logic
        Ok(())
    }
}

/// Configuration constants
pub const CONFIG: Config = Config {
    api_version: "v1",
    max_retries: 3,
    timeout_ms: 5000,
    endpoints: Endpoints {
        auth: "/auth",
        users: "/users",
        posts: "/posts",
    },
};

pub struct Config {
    api_version: &'static str,
    max_retries: u32,
    timeout_ms: u64,
    endpoints: Endpoints,
}

pub struct Endpoints {
    auth: &'static str,
    users: &'static str,
    posts: &'static str,
}

/// Generic event emitter implementation
pub struct EventEmitter<T: ToString> {
    listeners: std::collections::HashMap<T, Vec<Box<dyn Fn()>>>,
}

impl<T: ToString + std::cmp::Eq + std::hash::Hash> EventEmitter<T> {
    pub fn new() -> Self {
        Self {
            listeners: std::collections::HashMap::new(),
        }
    }

    pub fn on<F>(&mut self, event: T, callback: F)
    where
        F: Fn() + 'static,
    {
        self.listeners
            .entry(event)
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }

    pub fn emit(&self, event: T) {
        if let Some(callbacks) = self.listeners.get(&event) {
            for callback in callbacks {
                callback();
            }
        }
    }
}

/// Generic repository trait
pub trait Repository<T> {
    fn create(&mut self, item: T) -> Result<T, Box<dyn std::error::Error>>;
    fn update(&mut self, id: u64, item: T) -> Result<T, Box<dyn std::error::Error>>;
    fn delete(&mut self, id: u64) -> Result<bool, Box<dyn std::error::Error>>;
    fn find_by_id(&self, id: u64) -> Result<Option<&T>, Box<dyn std::error::Error>>;
}

/// Generic binary tree node
pub struct TreeNode<T> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

impl<T> TreeNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }
}

/// Generic queue implementation
pub struct Queue<T> {
    items: Vec<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn enqueue(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.items.remove(0))
        }
    }

    pub fn peek(&self) -> Option<&T> {
        self.items.first()
    }
}

/// Observable pattern implementation
pub struct Observable<T> {
    value: T,
    subscribers: Vec<Box<dyn Fn(&T)>>,
}

impl<T: Clone> Observable<T> {
    pub fn new(initial_value: T) -> Self {
        Self {
            value: initial_value,
            subscribers: Vec::new(),
        }
    }

    pub fn subscribe<F>(&mut self, callback: F)
    where
        F: Fn(&T) + 'static,
    {
        self.subscribers.push(Box::new(callback));
    }

    pub fn next(&mut self, value: T) {
        self.value = value.clone();
        for callback in &self.subscribers {
            callback(&self.value);
        }
    }
}

// Helper function for debouncing
pub fn debounce<F, T>(mut callback: F, duration: std::time::Duration) -> impl FnMut(T)
where
    F: FnMut(T),
    T: Clone + Send + 'static,
{
    let mut timeout_handle: Option<tokio::task::JoinHandle<()>> = None;

    move |arg: T| {
        if let Some(handle) = timeout_handle.take() {
            handle.abort();
        }

        let arg = arg.clone();
        timeout_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            callback(arg);
        }));
    }
}
