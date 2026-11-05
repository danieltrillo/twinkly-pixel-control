"""Fast pixel layout conversion using strided slice copies (no per-pixel loop)."""

from __future__ import annotations


def convert(src: bytes | bytearray, src_layout: str, dst_layout: str, pixels: int) -> bytearray:
    """Reorder channels, e.g. console RGBW -> Twinkly WRGB.

    Channels in dst that src does not have (W from an RGB console) are zero.
    Channels in src that dst does not have (W to an RGB strand) are dropped.
    """
    sn, dn = len(src_layout), len(dst_layout)
    out = bytearray(pixels * dn)
    view = memoryview(src)[: pixels * sn]
    for di, letter in enumerate(dst_layout):
        si = src_layout.find(letter)
        if si >= 0:
            out[di::dn] = view[si::sn]
    return out
