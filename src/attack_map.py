"""
attack_map.py — Maps each NSL-KDD attack name to its category.

Categories: normal, dos, probe, r2l, u2r
Includes all attacks in both KDDTrain+ and KDDTest+ (test has novel types).
"""

# ---------- Training set attacks ----------
TRAIN_ATTACKS = {
    # Normal
    "normal": "normal",
    # DoS
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    # Probe
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    # R2L
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
}

# ---------- Test-only attacks (novel, not in training) ----------
TEST_ONLY_ATTACKS = {
    # DoS
    "apache2": "dos",
    "udpstorm": "dos",
    "processtable": "dos",
    "worm": "dos",
    "mailbomb": "dos",
    # Probe
    "mscan": "probe",
    "saint": "probe",
    # R2L
    "named": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "httptunnel": "r2l",
    # U2R
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}

# ---------- Combined mapping (used everywhere) ----------
ATTACK_MAP: dict[str, str] = {**TRAIN_ATTACKS, **TEST_ONLY_ATTACKS}


def get_category(attack_name: str) -> str:
    """Map an attack name to its category. Raises KeyError if unknown."""
    return ATTACK_MAP[attack_name.strip().lower()]


def get_binary_label(attack_name: str) -> int:
    """0 = normal, 1 = attack."""
    return 0 if attack_name.strip().lower() == "normal" else 1


def get_novel_attacks() -> list[str]:
    """Return attack names that appear only in the test set."""
    return list(TEST_ONLY_ATTACKS.keys())
