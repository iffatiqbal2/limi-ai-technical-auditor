NETWORK DESIGN FOR LIMI HUBS

Topology:
- 3 Limi Hubs connected to 1 Central Server
- All devices in same VLAN

IP Addressing:
Hub 1: 192.168.1.101/24
Hub 2: 192.168.1.102/24  
Hub 3: 192.168.1.103/24
Server: 192.168.1.100/24

Configuration (Huawei eNSP):
vlan batch 10
interface Vlanif10
 ip address 192.168.1.1 255.255.255.0
interface GigabitEthernet0/0/1
 port link-type access
 port default vlan 10