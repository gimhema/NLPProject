use crate::net_tcp;
use crate::net_event_listner;

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PacketType {
    Connect,
    Disconnect,
    Data,
}


#[derive(Debug, Clone)]
pub struct DataPacket {
    pub packet_type: u16,
    pub data: String,
}

impl DataPacket {
    pub fn new_unassigned(_packet_type : u16, _data : String) -> Self {
        Self { packet_type: _packet_type, data: _data }
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.data.len() > u16::MAX as usize { return Err("message too long"); }
        Ok(())
    }
}