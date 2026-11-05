from twinklybridge import mapcode


def test_codewords_roundtrip_and_unique():
    for n in (1, 2, 7, 240, 1440, 2048):
        words = [mapcode.codeword(i, n) for i in range(n)]
        assert len(set(words)) == n
        assert all(w < (1 << mapcode.word_bits(n)) for w in words)
        assert [mapcode.decode_word(w, n) for w in words] == list(range(n))


def test_neighbours_differ_in_one_index_bit():
    n = 1440
    for i in range(n - 1):
        a = mapcode.codeword(i, n) >> mapcode.CRC_BITS
        b = mapcode.codeword(i + 1, n) >> mapcode.CRC_BITS
        assert bin(a ^ b).count("1") == 1


def test_crc_rejects_most_single_and_double_errors():
    n = 1440
    nb = mapcode.word_bits(n)
    bad = total = 0
    for i in range(0, n, 37):
        w = mapcode.codeword(i, n)
        for x in range(nb):
            for y in range(x, nb):
                flipped = w ^ (1 << x) ^ (1 << y) if x != y else w ^ (1 << x)
                total += 1
                bad += mapcode.decode_word(flipped, n) is not None
    assert bad == 0  # x^4+x+1 catches every 1- and 2-bit error at this length


def test_table_matches_decode_word():
    n = 300
    table = mapcode.decode_table(n)
    for w, v in enumerate(table):
        got = mapcode.decode_word(w, n)
        assert (v if v >= 0 else None) == got


def test_schedule_shape_and_balance():
    n = 1440
    s = mapcode.schedule(n)
    bits = mapcode.word_bits(n)
    assert len(s) == mapcode.LEAD + 2 * mapcode.MARK + 2 + 2 * bits + mapcode.TAIL
    words = [mapcode.codeword(i, n) for i in range(n)]
    data = [sl for sl in s if sl[0] == "bit"]
    assert len(data) == mapcode.data_slots(n)
    for k in range(0, len(data), 2):
        a = mapcode.lit_mask(data[k], n, words)
        b = mapcode.lit_mask(data[k + 1], n, words)
        assert all(x != y for x, y in zip(a, b, strict=True))  # every pixel: bit then inverse
    assert mapcode.scan_seconds(n) == len(s) * mapcode.DEFAULT_SLOT
