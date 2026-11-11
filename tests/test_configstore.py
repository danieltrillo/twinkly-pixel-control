"""The config file as the setup page edits it."""

import os
import pathlib
import stat

import pytest
import yaml

from twinklybridge import configstore
from twinklybridge.config import ConfigError, parse_config

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "twinklybridge" / "example-config.yaml"


def raw():
    return {
        "input": {"cidr": "10.0.0.2/24"},
        "defaults": {"fps": 30},
        "devices": [{"name": "Tree 1", "ip": "10.0.0.5", "universe": "0.0.0", "pixels": 300}],
    }


def test_dump_round_trips_the_example():
    original = yaml.safe_load(EXAMPLE.read_text())
    text = configstore.dump(original)
    assert parse_config(yaml.safe_load(text)) == parse_config(original)
    assert "universe: 0.0.0" in text and "universe: 0.0.12" in text  # Art-Net universes stay readable
    assert text.startswith("# Twinkly Bridge configuration")
    # the layout is stable: dumping again gives the same file (bar the timestamp line)
    again = configstore.dump(yaml.safe_load(text))
    assert text.splitlines()[2:] == again.splitlines()[2:]


def test_add_devices_assigns_the_next_free_universes():
    r = raw()
    added = configstore.add_devices(
        r,
        [
            {"name": "Tree 2", "ip": "10.0.0.6", "pixels": "1500", "fps": "", "enabled": "true"},
            {"name": "Arch", "ip": "10.0.0.7", "pixels": 170, "input": "rgb"},
        ],
    )
    # Tree 1 is 300 RGBW px = 3 universes (0.0.0-0.0.2); Tree 2 is 12 more; Arch is RGB, 1 universe
    assert [d["universe"] for d in added] == ["0.0.3", "0.0.15"]
    assert added[0]["pixels"] == 1500 and added[0]["enabled"] is True and "fps" not in added[0]
    assert added[1]["input"] == "RGB"
    assert list(added[0]) == ["name", "ip", "universe", "pixels", "enabled"]  # tidy key order
    c = parse_config(r)
    assert c.devices[1].universe_label == "0.0.3 - 0.0.14" and c.devices[2].universe_label == "0.0.15"


def test_universe_is_canonical_per_protocol():
    r = raw()
    configstore.add_devices(r, [{"name": "B", "ip": "10.0.0.6", "universe": "17", "pixels": 10}])
    assert r["devices"][-1]["universe"] == "0.1.1"
    r["input"]["sacn"] = True
    configstore.add_devices(r, [{"name": "C", "ip": "10.0.0.7", "universe": "7", "pixels": 10, "protocol": "sacn"}])
    assert r["devices"][-1]["universe"] == 7
    assert configstore.next_universe(r, "sacn") == 8
    with pytest.raises(ConfigError, match="universe"):
        configstore.add_devices(r, [{"name": "D", "ip": "10.0.0.8", "universe": "0.0", "pixels": 10}])


def test_replace_patch_and_remove():
    r = raw()
    configstore.add_devices(r, [{"name": "Tree 2", "ip": "10.0.0.6", "pixels": 10}])
    dev = configstore.replace_device(r, "Tree 1", {"name": "Big tree", "ip": "10.0.0.5", "pixels": 600})
    assert r["devices"][0] is dev and dev["universe"] == "0.0.4"  # first free after Tree 2 (0.0.3)
    patched = configstore.patch_device(r, "Tree 2", {"enabled": "false", "fps": "20"})
    assert patched["enabled"] is False and patched["fps"] == 20
    configstore.patch_device(r, "Tree 2", {"fps": ""})
    assert "fps" not in r["devices"][1]
    configstore.remove_device(r, "Big tree")
    assert [d["name"] for d in r["devices"]] == ["Tree 2"]
    with pytest.raises(ConfigError, match="no device named"):
        configstore.remove_device(r, "Big tree")


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"name": "X", "ip": "nope", "pixels": 10}, "not an IPv4 address"),
        ({"name": "X", "ip": "10.0.0.9", "pixels": "lots"}, "device.pixels: expected a whole number"),
        ({"name": "X", "ip": "10.0.0.9", "pixels": 10, "colour": "red"}, "unknown key 'colour'"),
        ({"name": "Tree 1", "ip": "10.0.0.9", "pixels": 10}, "used twice"),
        ({"name": "X", "ip": "10.0.0.5", "pixels": 10}, "already used"),
    ],
)
def test_bad_devices_are_refused_with_the_key(spec, message):
    with pytest.raises(ConfigError, match=message):
        configstore.add_devices(raw(), [spec])


def test_set_sections():
    r = raw()
    r["input"] = {"ip": "10.0.0.2"}
    configstore.set_sections(
        r,
        {
            "input": {"cidr": "10.0.0.2/24", "artnet": "true", "sacn": "on", "reply": "", "short_name": "Bridge"},
            "defaults": {"fps": "25.5", "order": "wrgb", "on_loss": "black"},
            "log_level": "debug",
            "mapping": {"brightness": "", "keep_clips": "false"},
        },
    )
    assert r["input"] == {"cidr": "10.0.0.2/24", "artnet": True, "sacn": True, "short_name": "Bridge"}
    assert r["defaults"] == {"fps": 25.5, "order": "WRGB", "on_loss": "black"}
    assert r["log_level"] == "debug" and r["mapping"] == {"keep_clips": False}
    configstore.set_sections(r, {"mapping": {}})
    assert "mapping" not in r
    with pytest.raises(ConfigError, match="unknown section"):
        configstore.set_sections(r, {"colours": {}})
    with pytest.raises(ConfigError, match="defaults.fps"):
        configstore.set_sections(r, {"defaults": {"fps": "fast"}})


def test_describe_explains_each_device():
    d = configstore.describe(raw(), "/nowhere/config.yaml")
    assert d["error"] == "" and d["devices"][0]["derived"]["universes"] == "0.0.0 - 0.0.2"
    assert d["devices"][0]["derived"]["fps"] == 30 and d["next_universe"] == {"artnet": "0.0.3", "sacn": 1}
    broken = raw()
    broken["devices"][0]["pixels"] = "many"
    d = configstore.describe(broken, "/nowhere/config.yaml")
    assert "pixels" in d["error"] and d["devices"][0]["derived"] is None


def test_write_text_keeps_backups_and_mode(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n")
    os.chmod(path, 0o640)
    for i in range(configstore.BACKUPS + 2):
        backup = configstore.write_text(str(path), f"a: {i + 2}\n")
        assert backup.endswith(".bak") and os.path.exists(backup)
    assert path.read_text() == f"a: {configstore.BACKUPS + 3}\n"
    assert len(configstore.backups_of(str(path))) == configstore.BACKUPS
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    assert not os.path.exists(str(path) + ".tmp")
    assert configstore.load_raw(str(path)) == {"a": configstore.BACKUPS + 3}
    assert configstore.write_text(str(tmp_path / "new.yaml"), "b: 1\n") == ""  # nothing to back up


def test_write_problem(tmp_path):
    path = tmp_path / "config.yaml"
    assert configstore.write_problem(str(path)) == ""
    if os.geteuid() == 0:
        pytest.skip("root can write anything")
    os.chmod(tmp_path, 0o500)
    try:
        assert "not writable" in configstore.write_problem(str(path))
    finally:
        os.chmod(tmp_path, 0o700)


def test_raw_from_text_errors():
    with pytest.raises(ConfigError, match="line 2"):
        configstore.raw_from_text("a: 1\n  b: [\n")
    with pytest.raises(ConfigError, match="mapping"):
        configstore.raw_from_text("- just a list\n")
    assert configstore.raw_from_text("") == {}
