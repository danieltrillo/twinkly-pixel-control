"""Blink code for camera pixel mapping.

Every pixel gets a codeword: the Gray code of its index followed by a 4-bit
CRC. The scan shows each codeword bit as two time slots, the bit then its
inverse (Manchester style), so the decoder only ever compares a spot with
itself a moment later. That makes it immune to how bright a pixel looks on
camera, to background light, and to the camera's white balance.

    black x LEAD | ON x MARK | off | b0 ~b0 b1 ~b1 ... | off | ON x MARK | black x TAIL

The two ON marks are full flashes. The decoder finds the end of the first
and the start of the second in the video and spaces the data slots evenly
between them, so it does not have to trust the phone's frame rate or the
streamer's timing, only that slots are evenly spaced.

Gray code means neighbouring pixel numbers differ by one bit, so the glow
between two touching pixels decodes to one of the two instead of garbage.
The CRC throws out most of what is left over.

Pure standard library: the streamer (tools/twinkly_probe.py) runs this on
the bridge box and the decoder (tools/pixelmap.py) runs it next to OpenCV.
"""

from __future__ import annotations

CRC_BITS = 4
CRC_POLY = 0b10011  # x^4 + x + 1
LEAD = 3  # black slots before the first mark
MARK = 4  # full-on slots in each mark
TAIL = 2  # black slots after the last mark
DEFAULT_SLOT = 0.4  # seconds per slot: 12 frames at 30 fps, plenty of margin


def index_bits(n_pixels: int) -> int:
    """Bits needed for pixel indices 0 .. n-1."""
    return max(1, (max(n_pixels, 1) - 1).bit_length())


def word_bits(n_pixels: int) -> int:
    return index_bits(n_pixels) + CRC_BITS


def gray(i: int) -> int:
    return i ^ (i >> 1)


def ungray(g: int) -> int:
    i = 0
    while g:
        i ^= g
        g >>= 1
    return i


def crc4(value: int, nbits: int) -> int:
    reg = value << CRC_BITS
    for bit in range(nbits + CRC_BITS - 1, CRC_BITS - 1, -1):
        if reg & (1 << bit):
            reg ^= CRC_POLY << (bit - CRC_BITS)
    return reg & ((1 << CRC_BITS) - 1)


def codeword(index: int, n_pixels: int) -> int:
    nb = index_bits(n_pixels)
    g = gray(index)
    return (g << CRC_BITS) | crc4(g, nb)


def decode_word(word: int, n_pixels: int) -> int | None:
    """Pixel index for a codeword, or None if the CRC fails or it is out of range."""
    nb = index_bits(n_pixels)
    g = word >> CRC_BITS
    if g >= (1 << nb) or crc4(g, nb) != word & ((1 << CRC_BITS) - 1):
        return None
    i = ungray(g)
    return i if i < n_pixels else None


def decode_table(n_pixels: int) -> list[int]:
    """table[word] = pixel index, or -1 for an invalid word."""
    out = [-1] * (1 << word_bits(n_pixels))
    for i in range(n_pixels):
        out[codeword(i, n_pixels)] = i
    return out


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


def schedule(n_pixels: int) -> list[tuple]:
    """The slots of a scan, in order.

    ("black",) all off; ("on",) all on; ("bit", b, phase): pixel lit when bit b
    of its codeword equals phase (1 then 0).
    """
    slots: list[tuple] = [("black",)] * LEAD + [("on",)] * MARK + [("black",)]
    for b in range(word_bits(n_pixels)):
        slots.append(("bit", b, 1))
        slots.append(("bit", b, 0))
    slots += [("black",)] + [("on",)] * MARK + [("black",)] * TAIL
    return slots


def data_slots(n_pixels: int) -> int:
    return 2 * word_bits(n_pixels)


def scan_seconds(n_pixels: int, slot: float = DEFAULT_SLOT) -> float:
    return len(schedule(n_pixels)) * slot


def lit_mask(slot: tuple, n_pixels: int, words: list[int] | None = None) -> list[bool]:
    """Which pixels are lit during one slot."""
    kind = slot[0]
    if kind == "black":
        return [False] * n_pixels
    if kind == "on":
        return [True] * n_pixels
    _, b, phase = slot
    words = words if words is not None else [codeword(i, n_pixels) for i in range(n_pixels)]
    return [((w >> b) & 1) == phase for w in words]
