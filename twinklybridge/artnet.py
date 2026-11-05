"""Art-Net 4 packet encoding and decoding.

Only the opcodes ArtBridge needs are implemented in full. Everything else is
recognised by opcode so it can be counted or passed through untouched.

Art-Net(TM) Designed by and Copyright Artistic Licence Holdings Ltd.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

ARTNET_PORT = 6454
ARTNET_ID = b"Art-Net\x00"
PROTOCOL_VERSION = 14

OP_POLL = 0x2000
OP_POLL_REPLY = 0x2100
OP_DIAG_DATA = 0x2300
OP_COMMAND = 0x2400
OP_DMX = 0x5000
OP_NZS = 0x5100
OP_SYNC = 0x5200
OP_ADDRESS = 0x6000
OP_INPUT = 0x7000
OP_TOD_REQUEST = 0x8000
OP_TOD_DATA = 0x8100
OP_TOD_CONTROL = 0x8200
OP_RDM = 0x8300
OP_RDM_SUB = 0x8400
OP_TIME_CODE = 0x9700
OP_TRIGGER = 0x9900
OP_IP_PROG = 0xF800
OP_IP_PROG_REPLY = 0xF900

OPCODE_NAMES = {
    OP_POLL: "ArtPoll",
    OP_POLL_REPLY: "ArtPollReply",
    OP_DIAG_DATA: "ArtDiagData",
    OP_COMMAND: "ArtCommand",
    OP_DMX: "ArtDmx",
    OP_NZS: "ArtNzs",
    OP_SYNC: "ArtSync",
    OP_ADDRESS: "ArtAddress",
    OP_INPUT: "ArtInput",
    OP_TOD_REQUEST: "ArtTodRequest",
    OP_TOD_DATA: "ArtTodData",
    OP_TOD_CONTROL: "ArtTodControl",
    OP_RDM: "ArtRdm",
    OP_RDM_SUB: "ArtRdmSub",
    OP_TIME_CODE: "ArtTimeCode",
    OP_TRIGGER: "ArtTrigger",
    OP_IP_PROG: "ArtIpProg",
    OP_IP_PROG_REPLY: "ArtIpProgReply",
}

# Node "Style" codes used in ArtPollReply.
STYLE_NODE = 0x00
STYLE_CONTROLLER = 0x01
STYLE_MEDIA = 0x02
STYLE_ROUTE = 0x03
STYLE_BACKUP = 0x04
STYLE_CONFIG = 0x05
STYLE_VISUAL = 0x06

STYLE_NAMES = {
    STYLE_NODE: "Node",
    STYLE_CONTROLLER: "Controller",
    STYLE_MEDIA: "Media server",
    STYLE_ROUTE: "Router",
    STYLE_BACKUP: "Backup",
    STYLE_CONFIG: "Config tool",
    STYLE_VISUAL: "Visualiser",
}

# PortTypes bits. "Output" means the node outputs DMX *from* Art-Net, i.e. it
# consumes ArtDmx. "Input" means the node puts DMX *onto* Art-Net.
PORT_TYPE_OUTPUT = 0x80
PORT_TYPE_INPUT = 0x40

# ArtPoll flag bits.
POLL_FLAG_REPLY_ON_CHANGE = 0x02
POLL_FLAG_DIAGNOSTICS = 0x04
POLL_FLAG_DIAG_UNICAST = 0x08
POLL_FLAG_DISABLE_VLC = 0x10
POLL_FLAG_TARGETED = 0x20

# GoodInput / GoodOutput bits.
GOOD_INPUT_DATA_RECEIVED = 0x80
GOOD_OUTPUT_DATA_TRANSMITTED = 0x80

# Status2 bits.
STATUS2_WEB_CONFIG = 0x01
STATUS2_DHCP_CAPABLE = 0x04
STATUS2_PORT_ADDRESS_15BIT = 0x08
STATUS2_SACN_CAPABLE = 0x10

MAX_PORT_ADDRESS = 0x7FFF
MAX_DMX_SLOTS = 512


def opcode(data: bytes) -> int | None:
    """Return the opcode of an Art-Net packet, or None if it is not Art-Net."""
    if len(data) < 10 or data[:8] != ARTNET_ID:
        return None
    return data[8] | (data[9] << 8)


def opcode_name(op: int) -> str:
    return OPCODE_NAMES.get(op, f"Op0x{op:04X}")


# --------------------------------------------------------------------------
# Port-Address helpers (15 bit: Net 7 bits, Sub-Net 4 bits, Universe 4 bits)
# --------------------------------------------------------------------------


def port_address(net: int, sub: int, uni: int) -> int:
    if not (0 <= net <= 127 and 0 <= sub <= 15 and 0 <= uni <= 15):
        raise ValueError(f"Art-Net address {net}.{sub}.{uni} is out of range (net 0-127, subnet 0-15, universe 0-15)")
    return (net << 8) | (sub << 4) | uni


def split_port_address(pa: int) -> tuple[int, int, int]:
    return (pa >> 8) & 0x7F, (pa >> 4) & 0x0F, pa & 0x0F


def format_universe(pa: int) -> str:
    """Render a Port-Address as net.subnet.universe, e.g. 0.0.1."""
    net, sub, uni = split_port_address(pa)
    return f"{net}.{sub}.{uni}"


def parse_universe(value: object) -> int:
    """Parse an Art-Net universe written as a Port-Address number (0-32767)
    or as net.subnet.universe (for example "0.0.1")."""
    if isinstance(value, bool):
        raise ValueError("universe must be a number or net.subnet.universe")
    if isinstance(value, int):
        if not 0 <= value <= MAX_PORT_ADDRESS:
            raise ValueError(f"Art-Net universe {value} is out of range (0-32767)")
        return value
    if isinstance(value, float):
        raise ValueError(f"universe {value!r} is ambiguous; write three numbers like 0.0.1 or a single whole number")
    if isinstance(value, str):
        text = value.strip()
        parts = text.split(".")
        try:
            if len(parts) == 3:
                return port_address(*(int(p) for p in parts))
            if len(parts) == 1:
                return parse_universe(int(text))
        except ValueError as exc:
            if "out of range" in str(exc):
                raise
        raise ValueError(f"universe {value!r} is not valid; use net.subnet.universe like 0.0.1 or a whole number")
    raise ValueError(f"universe {value!r} is not valid")


# --------------------------------------------------------------------------
# ArtDmx / ArtNzs
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ArtDmx:
    universe: int
    data: bytes
    sequence: int = 0
    physical: int = 0
    start_code: int = 0


def decode_dmx(data: bytes) -> ArtDmx | None:
    """Decode an ArtDmx or ArtNzs packet. Returns None if not one of those."""
    if len(data) < 18:
        return None
    op = opcode(data)
    if op == OP_DMX:
        start_code = 0
        physical = data[13]
    elif op == OP_NZS:
        start_code = data[13]
        physical = 0
    else:
        return None
    length = (data[16] << 8) | data[17]
    if length < 1:
        return None
    payload = data[18 : 18 + min(length, MAX_DMX_SLOTS)]
    if not payload:
        return None
    universe = data[14] | ((data[15] & 0x7F) << 8)
    return ArtDmx(
        universe=universe,
        data=bytes(payload),
        sequence=data[12],
        physical=physical,
        start_code=start_code,
    )


_DMX_HEADER = struct.Struct("<8sHBBBBBBBB")


def encode_dmx(
    universe: int,
    data: bytes,
    sequence: int = 0,
    physical: int = 0,
    start_code: int = 0,
) -> bytes:
    """Build an ArtDmx packet (or ArtNzs when start_code is not zero)."""
    if len(data) > MAX_DMX_SLOTS:
        data = data[:MAX_DMX_SLOTS]
    if len(data) < 2 or len(data) % 2:
        # Art-Net wants an even length between 2 and 512.
        data = bytes(data) + b"\x00" * (2 - len(data) if len(data) < 2 else 1)
    op = OP_DMX if start_code == 0 else OP_NZS
    byte13 = physical & 0xFF if op == OP_DMX else start_code & 0xFF
    header = _DMX_HEADER.pack(
        ARTNET_ID,
        op,
        0,
        PROTOCOL_VERSION,
        sequence & 0xFF,
        byte13,
        universe & 0xFF,
        (universe >> 8) & 0x7F,
        len(data) >> 8,
        len(data) & 0xFF,
    )
    return header + bytes(data)


# --------------------------------------------------------------------------
# ArtSync
# --------------------------------------------------------------------------


def encode_sync() -> bytes:
    return struct.pack("<8sHBBBB", ARTNET_ID, OP_SYNC, 0, PROTOCOL_VERSION, 0, 0)


# --------------------------------------------------------------------------
# ArtPoll
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ArtPoll:
    flags: int = 0
    diag_priority: int = 0
    target_top: int | None = None
    target_bottom: int | None = None

    def wants(self, pa: int) -> bool:
        """True if this poll is interested in the given Port-Address."""
        if self.target_top is None or self.target_bottom is None:
            return True
        return self.target_bottom <= pa <= self.target_top


def decode_poll(data: bytes) -> ArtPoll | None:
    if opcode(data) != OP_POLL or len(data) < 12:
        return None
    flags = data[12] if len(data) > 12 else 0
    diag = data[13] if len(data) > 13 else 0
    poll = ArtPoll(flags=flags, diag_priority=diag)
    if flags & POLL_FLAG_TARGETED and len(data) >= 18:
        poll.target_top = (data[14] << 8) | data[15]
        poll.target_bottom = (data[16] << 8) | data[17]
    return poll


def encode_poll(flags: int = POLL_FLAG_REPLY_ON_CHANGE, diag_priority: int = 0x10) -> bytes:
    """A minimal 14 byte ArtPoll, which every node understands."""
    return struct.pack("<8sHBBBB", ARTNET_ID, OP_POLL, 0, PROTOCOL_VERSION, flags, diag_priority)


# --------------------------------------------------------------------------
# ArtPollReply
# --------------------------------------------------------------------------

_REPLY = struct.Struct(
    "<8sH"  # id, opcode
    "4sH"  # ip, port (little endian 0x1936)
    "BB"  # VersInfoH, VersInfoL
    "BB"  # NetSwitch, SubSwitch
    "BB"  # OemHi, OemLo
    "B"  # UbeaVersion
    "B"  # Status1
    "BB"  # EstaManLo, EstaManHi
    "18s"  # PortName
    "64s"  # LongName
    "64s"  # NodeReport
    "BB"  # NumPortsHi, NumPortsLo
    "4s4s4s4s4s"  # PortTypes, GoodInput, GoodOutput, SwIn, SwOut
    "BBB"  # AcnPriority, SwMacro, SwRemote
    "3s"  # Spare
    "B"  # Style
    "6s"  # MAC
    "4s"  # BindIp
    "B"  # BindIndex
    "B"  # Status2
    "4s"  # GoodOutputB
    "B"  # Status3
    "6s"  # DefaultRespUID
    "BB"  # UserHi, UserLo
    "BB"  # RefreshRateHi, RefreshRateLo
    "B"  # BackgroundQueuePolicy
    "10s"  # Filler
)
REPLY_LENGTH = _REPLY.size
assert REPLY_LENGTH == 239


def _cstr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _ip_str(raw: bytes) -> str:
    return ".".join(str(b) for b in raw[:4])


def _ip_bytes(ip: str) -> bytes:
    parts = ip.split(".")
    if len(parts) != 4:
        raise ValueError(f"bad IPv4 address {ip!r}")
    return bytes(int(p) for p in parts)


@dataclass(slots=True)
class ArtPollReply:
    ip: str = "0.0.0.0"
    port: int = ARTNET_PORT
    version: int = 0
    net_switch: int = 0
    sub_switch: int = 0
    oem: int = 0
    ubea_version: int = 0
    status1: int = 0
    esta_man: int = 0
    short_name: str = ""
    long_name: str = ""
    node_report: str = ""
    num_ports: int = 0
    port_types: bytes = b"\x00\x00\x00\x00"
    good_input: bytes = b"\x00\x00\x00\x00"
    good_output: bytes = b"\x00\x00\x00\x00"
    sw_in: bytes = b"\x00\x00\x00\x00"
    sw_out: bytes = b"\x00\x00\x00\x00"
    acn_priority: int = 0
    sw_macro: int = 0
    sw_remote: int = 0
    style: int = STYLE_NODE
    mac: bytes = b"\x00" * 6
    bind_ip: str = "0.0.0.0"
    bind_index: int = 0
    status2: int = 0
    good_output_b: bytes = b"\x00\x00\x00\x00"
    status3: int = 0

    def _port_count(self) -> int:
        return self.num_ports if 0 < self.num_ports <= 4 else 4

    def output_universes(self) -> list[int]:
        """Port-Addresses this node outputs to DMX (it consumes ArtDmx for these)."""
        return [
            port_address(self.net_switch & 0x7F, self.sub_switch & 0x0F, self.sw_out[i] & 0x0F)
            for i in range(self._port_count())
            if self.port_types[i] & PORT_TYPE_OUTPUT
        ]

    def input_universes(self) -> list[int]:
        """Port-Addresses this node puts onto Art-Net (it sends ArtDmx for these)."""
        return [
            port_address(self.net_switch & 0x7F, self.sub_switch & 0x0F, self.sw_in[i] & 0x0F)
            for i in range(self._port_count())
            if self.port_types[i] & PORT_TYPE_INPUT
        ]

    @property
    def mac_str(self) -> str:
        return ":".join(f"{b:02x}" for b in self.mac)

    @property
    def style_name(self) -> str:
        return STYLE_NAMES.get(self.style, f"Style 0x{self.style:02X}")


def decode_poll_reply(data: bytes) -> ArtPollReply | None:
    """Decode an ArtPollReply. Short packets from older nodes are accepted;
    missing trailing fields read as zero."""
    if opcode(data) != OP_POLL_REPLY or len(data) < 26:
        return None
    if len(data) < REPLY_LENGTH:
        data = bytes(data) + b"\x00" * (REPLY_LENGTH - len(data))
    f = _REPLY.unpack_from(data)
    return ArtPollReply(
        ip=_ip_str(f[2]),
        port=f[3],
        version=(f[4] << 8) | f[5],
        net_switch=f[6],
        sub_switch=f[7],
        oem=(f[8] << 8) | f[9],
        ubea_version=f[10],
        status1=f[11],
        esta_man=f[12] | (f[13] << 8),
        short_name=_cstr(f[14]),
        long_name=_cstr(f[15]),
        node_report=_cstr(f[16]),
        num_ports=(f[17] << 8) | f[18],
        port_types=f[19],
        good_input=f[20],
        good_output=f[21],
        sw_in=f[22],
        sw_out=f[23],
        acn_priority=f[24],
        sw_macro=f[25],
        sw_remote=f[26],
        style=f[28],
        mac=f[29],
        bind_ip=_ip_str(f[30]),
        bind_index=f[31],
        status2=f[32],
        good_output_b=f[33],
        status3=f[34],
    )


def _fixed(text: str, size: int) -> bytes:
    raw = text.encode("utf-8", "replace")[: size - 1]
    return raw + b"\x00" * (size - len(raw))


def encode_poll_reply(reply: ArtPollReply) -> bytes:
    """Build a full 239 byte Art-Net 4 ArtPollReply."""
    return _REPLY.pack(
        ARTNET_ID,
        OP_POLL_REPLY,
        _ip_bytes(reply.ip),
        reply.port,
        (reply.version >> 8) & 0xFF,
        reply.version & 0xFF,
        reply.net_switch & 0x7F,
        reply.sub_switch & 0x0F,
        (reply.oem >> 8) & 0xFF,
        reply.oem & 0xFF,
        reply.ubea_version & 0xFF,
        reply.status1 & 0xFF,
        reply.esta_man & 0xFF,
        (reply.esta_man >> 8) & 0xFF,
        _fixed(reply.short_name, 18),
        _fixed(reply.long_name, 64),
        _fixed(reply.node_report, 64),
        (reply.num_ports >> 8) & 0xFF,
        reply.num_ports & 0xFF,
        bytes(reply.port_types[:4]).ljust(4, b"\x00"),
        bytes(reply.good_input[:4]).ljust(4, b"\x00"),
        bytes(reply.good_output[:4]).ljust(4, b"\x00"),
        bytes(reply.sw_in[:4]).ljust(4, b"\x00"),
        bytes(reply.sw_out[:4]).ljust(4, b"\x00"),
        reply.acn_priority & 0xFF,
        reply.sw_macro & 0xFF,
        reply.sw_remote & 0xFF,
        b"\x00\x00\x00",
        reply.style & 0xFF,
        bytes(reply.mac[:6]).ljust(6, b"\x00"),
        _ip_bytes(reply.bind_ip),
        reply.bind_index & 0xFF,
        reply.status2 & 0xFF,
        bytes(reply.good_output_b[:4]).ljust(4, b"\x00"),
        reply.status3 & 0xFF,
        b"\x00" * 6,
        0,
        0,
        0,
        0,
        0,
        b"\x00" * 10,
    )


def node_report(code: int, counter: int, text: str) -> str:
    """Format a NodeReport string: '#0001 [0012] text'."""
    return f"#{code:04x} [{counter % 10000:04d}] {text}"


RC_POWER_OK = 0x0001
