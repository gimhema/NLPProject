use crate::net_tcp;
use crate::net_packet;


pub struct NetEventListener {
    tcp_listener: net_tcp::TcpListener,
}

impl NetEventListener {
    pub fn new(address: &str, port: u16) -> Self {
        NetEventListener {
            tcp_listener: net_tcp::TcpListener::new(address, port),
        }
    }

    pub fn start_listening(&mut self) {
        self.tcp_listener.start();
    }

    pub fn stop_listening(&mut self) {
        self.tcp_listener.stop();
    }

    pub fn read_packet(&mut self) -> Option<net_packet::DataPacket> {
        if let Some(_stream) = self.tcp_listener.listen() {
            // 여기서 실제로 패킷을 읽어오는 로직을 구현해야 함
            // 예시로 빈 패킷을 반환
            Some(net_packet::DataPacket::new_unassigned(0, String::new()))
        } else {
            None
        }
    }

    pub fn event_loop(&mut self) {
        loop {
            if let Some(packet) = self.read_packet() {
                println!("Received packet: {:?}", packet);
                // 패킷 처리 로직 구현
                self.event_action(&packet);
            }
        }
    }

    pub fn event_action(&mut self, packet: &net_packet::DataPacket) {
        // 패킷에 따른 이벤트 처리 로직 구현
        println!("Handling packet action for: {:?}", packet);
        
        let _recv_type = packet.packet_type;
        let _recv_data = &packet.data;

        match _recv_data {
            Connect=> {
                println!("Handling Connect action");
            }
            Disconnect => {
                println!("Handling Disconnect action");
            }
            Data => {
                println!("Handling Data action");
            }
            _ => {
                println!("No specific action for data: {}", _recv_data);
            }
        }
    }
}
