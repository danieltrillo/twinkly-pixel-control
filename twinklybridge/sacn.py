"""sACN (ANSI E1.31) packet encoding and decoding.

Covers E1.31 data packets and universe discovery packets. Synchronization
packets are recognised and ignored.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

SACN_PORT = 5568
ACN_PACKET_ID = b"ASC-E1.17\x00\x00\x00"

VECTOR_ROOT_E131_DATA = 0x00000004
VECTOR_ROOT_E131_EXTENDED = 0x00000008
VECTOR_E131_DATA_PACKET = 0x00000002
VECTOR_E131_EXTENDED_SYNCHRONIZATION = 0x00000001
VECTOR_E131_EXTENDED_DISCOVERY = 0x00000002
VECTOR_DMP_SET_PROPERTY = 0x02
VECTOR_UNIVERSE_DISCOVERY_UNIVERSE_LIST = 0x00000001

DISCOVERY_UNIVERSE = 64214
DISCOVERY_INTERVAL = 10.0

OPT_PREVIEW = 0x80
OPT_TERMINATED = 0x40
OPT_FORCE_SYNC = 0x20

MIN_UNIVERSE = 1
MAX_UNIVERSE = 63999
MAX_SLOTS = 512
DEFAULT_PRIORITY = 100

_DATA_HEADER_LEN = 126


def multicast_address(universe: int) -> str:
    return f"239.255.{(universe >> 8) & 0xFF}.{universe & 0xFF}"


def parse_universe(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("sACN universe must be a whole number between 1 and 63999")
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"sACN universe {value!r} must be a whole number between 1 and 63999")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError(f"sACN universe {value!r} must be a whole number between 1 and 63999")
    if not MIN_UNIVERSE <= value <= MAX_UNIVERSE:
        raise ValueError(f"sACN universe {value} is out of range (1-63999)")
    return value


@dataclass(slots=True)
class SacnData:
    universe: int
    data: bytes
    start_code: int = 0
    priority: int = DEFAULT_PRIORITY
    sequence: int = 0
    options: int = 0
    source_name: str = ""
    cid: bytes = b"\x00" * 16
    sync_address: int = 0

    @property
    def preview(self) -> bool:
        return bool(self.options & OPT_PREVIEW)

    @property
    def terminated(self) -> bool:
        return bool(self.options & OPT_TERMINATED)


def is_sacn(data: bytes) -> bool:
    return len(data) >= 38 and data[0:2] == b"\x00\x10" and data[4:16] == ACN_PACKET_ID


def root_vector(data: bytes) -> int | None:
    if not is_sacn(data):
        return None
    return struct.unpack_from(">I", data, 18)[0]


def decode_data(data: bytes) -> SacnData | None:
    """Decode an E1.31 data packet. Returns None for anything else."""
    if len(data) < _DATA_HEADER_LEN or not is_sacn(data):
        return None
    if struct.unpack_from(">I", data, 18)[0] != VECTOR_ROOT_E131_DATA:
        return None
    if struct.unpack_from(">I", data, 40)[0] != VECTOR_E131_DATA_PACKET:
        return None
    if data[117] != VECTOR_DMP_SET_PROPERTY or data[118] != 0xA1:
        return None
    count = struct.unpack_from(">H", data, 123)[0]
    if count < 1:
        return None
    slots = data[126 : 126 + min(count - 1, MAX_SLOTS)]
    return SacnData(
        universe=struct.unpack_from(">H", data, 113)[0],
        data=bytes(slots),
        start_code=data[125],
        priority=data[108],
        sequence=data[111],
        options=data[112],
        source_name=data[44:108].split(b"\x00", 1)[0].decode("utf-8", "replace"),
        cid=bytes(data[22:38]),
        sync_address=struct.unpack_from(">H", data, 109)[0],
    )


def _fixed(text: str, size: int) -> bytes:
    raw = text.encode("utf-8", "replace")[: size - 1]
    return raw + b"\x00" * (size - len(raw))


def _root_layer(cid: bytes, total: int, vector: int) -> bytes:
    return (
        b"\x00\x10\x00\x00"
        + ACN_PACKET_ID
        + struct.pack(">HI", 0x7000 | (total - 16), vector)
        + bytes(cid[:16]).ljust(16, b"\x00")
    )


def encode_data(
    universe: int,
    data: bytes,
    cid: bytes,
    source_name: str,
    priority: int = DEFAULT_PRIORITY,
    sequence: int = 0,
    start_code: int = 0,
    options: int = 0,
    sync_address: int = 0,
) -> bytes:
    """Build an E1.31 data packet."""
    if len(data) > MAX_SLOTS:
        data = data[:MAX_SLOTS]
    total = _DATA_HEADER_LEN + len(data)
    root = _root_layer(cid, total, VECTOR_ROOT_E131_DATA)
    framing = struct.pack(
        ">HI64sBHBBH",
        0x7000 | (total - 38),
        VECTOR_E131_DATA_PACKET,
        _fixed(source_name, 64),
        max(0, min(200, priority)),
        sync_address & 0xFFFF,
        sequence & 0xFF,
        options & 0xFF,
        universe & 0xFFFF,
    )
    dmp = struct.pack(
        ">HBBHHHB",
        0x7000 | (total - 115),
        VECTOR_DMP_SET_PROPERTY,
        0xA1,
        0x0000,
        0x0001,
        len(data) + 1,
        start_code & 0xFF,
    )
    return root + framing + dmp + bytes(data)


def encode_discovery(cid: bytes, source_name: str, universes: list[int], page: int = 0, last_page: int = 0) -> bytes:
    """Build an E1.31 universe discovery packet (max 512 universes per page)."""
    universes = sorted(universes)[:512]
    total = 120 + 2 * len(universes)
    root = _root_layer(cid, total, VECTOR_ROOT_E131_EXTENDED)
    framing = struct.pack(
        ">HI64s4s",
        0x7000 | (total - 38),
        VECTOR_E131_EXTENDED_DISCOVERY,
        _fixed(source_name, 64),
        b"\x00" * 4,
    )
    disc = struct.pack(
        ">HIBB",
        0x7000 | (total - 112),
        VECTOR_UNIVERSE_DISCOVERY_UNIVERSE_LIST,
        page & 0xFF,
        last_page & 0xFF,
    ) + b"".join(struct.pack(">H", u) for u in universes)
    return root + framing + disc
