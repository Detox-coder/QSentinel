import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 4433

CERTS_DIR = os.path.join(BASE_DIR, "certs")
CERT_FILE = os.path.join(CERTS_DIR, "cert.pem")
KEY_FILE = os.path.join(CERTS_DIR, "key.pem")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TRAFFIC_WEB_BROWSE = 0x01
TRAFFIC_STREAMING = 0x02
TRAFFIC_EXFIL_SLOW = 0x03
TRAFFIC_EXFIL_BURST = 0x04
TRAFFIC_C2_BEACON = 0x05
TRAFFIC_TUNNEL = 0x06
TRAFFIC_STEALTH_C2 = 0x07
TRAFFIC_HYBRID = 0x08

LABEL_MAP = {
    TRAFFIC_WEB_BROWSE: "benign_web",
    TRAFFIC_STREAMING: "benign_streaming",
    TRAFFIC_EXFIL_SLOW: "malicious_exfil_slow",
    TRAFFIC_EXFIL_BURST: "malicious_exfil_burst",
    TRAFFIC_C2_BEACON: "malicious_c2_beacon",
    TRAFFIC_TUNNEL: "malicious_tunnel",
    TRAFFIC_STEALTH_C2: "malicious_stealth_c2",
    TRAFFIC_HYBRID: "malicious_hybrid",
}

DEFAULT_FLOWS_PER_CLASS = 50
