# Generators package
from generators.benign import WebBrowsingGenerator, StreamingGenerator
from generators.malicious import (
    SlowExfilGenerator,
    BurstExfilGenerator,
    C2BeaconGenerator,
    TunnelGenerator,
    StealthC2Generator,
    HybridAttackGenerator,
)

ALL_GENERATORS = [
    WebBrowsingGenerator,
    StreamingGenerator,
    SlowExfilGenerator,
    BurstExfilGenerator,
    C2BeaconGenerator,
    TunnelGenerator,
    StealthC2Generator,
    HybridAttackGenerator,
]

__all__ = [
    "WebBrowsingGenerator",
    "StreamingGenerator",
    "SlowExfilGenerator",
    
    "BurstExfilGenerator",
    "C2BeaconGenerator",
    "TunnelGenerator",
    "StealthC2Generator",
    "HybridAttackGenerator",
    "ALL_GENERATORS",
]
