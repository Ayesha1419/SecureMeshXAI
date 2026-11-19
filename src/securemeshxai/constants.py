"""Canonical aligned features and class definitions."""
FEATURES = [
    'duration','orig_bytes','resp_bytes','orig_pkts','resp_pkts',
    'orig_ip_bytes','resp_ip_bytes','missed_bytes',
    'byte_ratio','mean_packet_size','packet_rate','mean_interarrival',
    'proto_tcp','proto_udp','proto_icmp',
    'state_success','state_s0','state_rej','state_reset','state_other',
    'forward_reliability','temporal_stability','neighbor_agreement','anomaly_persistence'
]
FLOW_FEATURES = FEATURES[:20]
TRUST_FEATURES = FEATURES[20:]
TON_CLASSES = ['normal','backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss']
