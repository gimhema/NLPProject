use std::net::{TcpListener as StdTcpListener, TcpStream};

pub struct TcpListener {
    pub address: String,
    pub port: u16,
    pub tcp_socket: Option<StdTcpListener>,
}

impl TcpListener {
    pub fn new(address: &str, port: u16) -> Self {
        TcpListener {
            address: address.to_string(),
            port,
            tcp_socket: None,
        }
    }

    pub fn start(&self) {
        // TCP 리스너 시작 로직 구현
        println!("Starting TCP listener on {}:{}", self.address, self.port);

        let listener = StdTcpListener::bind(format!("{}:{}", self.address, self.port))
            .expect("Could not bind TCP listener");
    }

    pub fn listen(&self) -> Option<TcpStream> {
        // 클라이언트 연결 수락 로직 구현
        if let Some(ref listener) = self.tcp_socket {
            match listener.accept() {
                Ok((stream, addr)) => {
                    println!("New connection from {}", addr);
                    Some(stream)
                }
                Err(e) => {
                    eprintln!("Failed to accept connection: {}", e);
                    None
                }
            }
        } else {
            eprintln!("TCP listener is not started.");
            None
        }
    }

    pub fn stop(&self) {
        // TCP 리스너 중지 로직 구현
        println!("Stopping TCP listener on {}:{}", self.address, self.port);
        // 실제로는 TcpListener를 닫는 로직이 필요함

        if let Some(ref listener) = self.tcp_socket {
            drop(listener);

        }
        else {
            eprintln!("TCP listener is not started.");
        }
    }
    
}


