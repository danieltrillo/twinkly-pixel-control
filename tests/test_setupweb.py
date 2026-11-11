"""The setup page's API: edit the config file from the browser and find controllers."""

import asyncio
import os

import pytest
from aiohttp.test_utils import TestClient, TestServer

from twinklybridge import web
from twinklybridge.app import Supervisor
from twinklybridge.config import load_config
from twinklybridge.testtools import FakeTwinkly
from twinklybridge.twinkly import Discovered
from twinklybridge.web import make_app

CONFIG = """# my notes
input: {cidr: 10.0.0.2/24}
devices:
  - {name: Tree 1, ip: 10.0.0.5, universe: 0.0.0, pixels: 80}
"""


def _supervisor(cfg_path):
    sup = Supervisor(str(cfg_path))
    sup.config = load_config(str(cfg_path))
    reloads = []

    async def fake_reload():  # the real one restarts the engine (sockets); here: re-read the file
        reloads.append(1)
        sup.config = load_config(str(cfg_path))
        sup.config_error = ""

    sup.reload = fake_reload
    return sup, reloads


def test_edit_devices_and_settings(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    sup, reloads = _supervisor(cfg_path)

    async def main():
        async with TestClient(TestServer(make_app(sup))) as client:
            r = await client.get("/setup")
            assert r.status == 200 and "text/html" in r.content_type

            s = await (await client.get("/api/setup/state")).json()
            assert s["error"] == "" and s["write_problem"] == "" and s["path"] == str(cfg_path)
            assert s["devices"][0]["derived"]["universes"] == "0.0.0" and s["next_universe"]["artnet"] == "0.0.1"

            # a bad device changes nothing
            before = cfg_path.read_text()
            r = await client.post("/api/setup/devices", json={"device": {"name": "X", "ip": "nope", "pixels": 10}})
            assert r.status == 400 and "IPv4" in (await r.json())["error"]
            assert cfg_path.read_text() == before and reloads == []

            # add one: universe assigned, file rewritten with a backup, bridge reloaded
            r = await client.post(
                "/api/setup/devices", json={"device": {"name": "Tree 2", "ip": "10.0.0.6", "pixels": "300", "fps": ""}}
            )
            j = await r.json()
            assert r.status == 200, j
            assert j["result"][0]["universe"] == "0.0.1" and j["backup"].endswith(".bak") and reloads == [1]
            assert [d["derived"]["universes"] for d in j["state"]["devices"]] == ["0.0.0", "0.0.1 - 0.0.3"]
            text = cfg_path.read_text()
            assert "name: Tree 2" in text and "universe: 0.0.1" in text and "# my notes" not in text
            assert os.path.exists(str(cfg_path) + "." + j["backup"].split(".")[-2] + ".bak")
            assert sup.config.devices[1].name == "Tree 2"

            # several at once (discovery's "add selected")
            r = await client.post(
                "/api/setup/devices",
                json={
                    "devices": [
                        {"name": "A", "ip": "10.0.0.7", "pixels": 128},
                        {"name": "B", "ip": "10.0.0.8", "pixels": 1},
                    ]
                },
            )
            j = await r.json()
            assert [d["universe"] for d in j["result"]] == ["0.0.4", "0.0.5"]

            # replace, patch, delete
            r = await client.put(
                "/api/setup/devices/Tree 1", json={"device": {"name": "Big", "ip": "10.0.0.5", "pixels": 10}}
            )
            j = await r.json()
            assert r.status == 200 and j["state"]["devices"][0]["spec"]["name"] == "Big"
            assert j["result"]["universe"] == "0.0.6"  # left blank: next free
            r = await client.patch("/api/setup/devices/Big", json={"enabled": "false"})
            j = await r.json()
            assert r.status == 200 and j["state"]["devices"][0]["derived"]["enabled"] is False
            assert "enabled: false" in cfg_path.read_text() and sup.config.devices[0].enabled is False
            r = await client.delete("/api/setup/devices/Big")
            assert r.status == 200
            r = await client.delete("/api/setup/devices/Big")
            assert r.status == 400 and "no device named" in (await r.json())["error"]
            assert [d.name for d in sup.config.devices] == ["Tree 2", "A", "B"]

            # settings sections
            r = await client.post(
                "/api/setup/settings",
                json={
                    "input": {"cidr": "10.0.0.2/24", "artnet": "true", "sacn": "true"},
                    "defaults": {"fps": "30"},
                    "log_level": "debug",
                },
            )
            j = await r.json()
            assert r.status == 200, j
            assert (
                j["state"]["input"]["sacn"] is True
                and j["state"]["defaults"] == {"fps": 30}
                and j["state"]["log_level"] == "debug"
            )
            assert sup.config.input.sacn and sup.config.devices[0].fps == 30 and sup.config.log_level == "debug"
            r = await client.post("/api/setup/settings", json={"defaults": {"fps": "fast"}})
            assert r.status == 400 and "defaults.fps" in (await r.json())["error"]

            # the raw file editor keeps the text as typed
            r = await client.post("/api/setup/yaml", json={"yaml": "input: {cidr: 10.0.0.2/24\n"})
            assert r.status == 400 and "YAML syntax error" in (await r.json())["error"]
            r = await client.post(
                "/api/setup/yaml",
                json={
                    "yaml": "# hand written\ninput: {cidr: 10.0.0.2/24}\ndevices: [{ip: 10.0.0.9, universe: 0.1.0, pixels: 5}]"
                },
            )
            assert r.status == 200, await r.text()
            assert cfg_path.read_text().startswith("# hand written\n") and sup.config.devices[0].ip == "10.0.0.9"
            s = await (await client.get("/api/setup/state")).json()
            assert s["devices"][0]["derived"]["universes"] == "0.1.0" and len(s["backups"]) >= 1

    asyncio.run(main())


def test_discover_and_look_up(tmp_path, monkeypatch):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    sup, _ = _supervisor(cfg_path)

    async def main():
        fake = FakeTwinkly(pixels=600, api="v2", host="127.0.0.1")
        await fake.start()
        monkeypatch.setattr(
            web, "discover", lambda targets, timeout: [Discovered("127.0.0.1", "TwPRO_fake", fake.http_port)]
        )
        try:
            async with TestClient(TestServer(make_app(sup))) as client:
                r = await client.post("/api/setup/discover", json={})
                j = await r.json()
                assert r.status == 200, j
                assert "255.255.255.255" in j["targets"] and "10.0.0.255" in j["targets"]
                (row,) = j["devices"]
                assert row["ip"] == "127.0.0.1" and row["name"] == "TwPRO_fake" and row["configured_as"] is None
                assert row["info"]["number_of_led"] == 600 and row["info"]["led_profile"] == "RGBW"
                assert row["info"]["api"] == "v2" and row["error"] == "" and fake.logins == 0  # no login: read only

                r = await client.post("/api/setup/discover", json={"ip": "127.0.0.1", "port": fake.http_port})
                j = await r.json()
                assert j["targets"] == [] and j["devices"][0]["info"]["number_of_led"] == 600
                r = await client.post("/api/setup/discover", json={"ip": "127.0.0.1", "port": fake.http_port + 1})
                j = await r.json()
                assert j["devices"][0]["info"] is None and j["devices"][0]["error"]
                r = await client.post("/api/setup/discover", json={"ip": "nope"})
                assert r.status == 400

                r = await client.post(
                    "/api/setup/devices", json={"device": {"name": "Found", "ip": "127.0.0.1", "pixels": 600}}
                )
                assert r.status == 200
                r = await client.post("/api/setup/discover", json={})
                assert (await r.json())["devices"][0]["configured_as"] == "Found"
        finally:
            await fake.stop()

    asyncio.run(main())


def test_discovery_targets_end_with_the_global_broadcast(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    sup, _ = _supervisor(cfg_path)
    targets = web._discovery_targets(sup, "10.9.9.255")
    assert targets[0] == "10.9.9.255" and targets[-1] == "255.255.255.255" and "10.0.0.255" in targets
    assert len(targets) == len(set(targets))


@pytest.mark.parametrize("body", [{"device": "x"}, {"devices": []}, {}])
def test_bad_bodies(tmp_path, body):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    sup, _ = _supervisor(cfg_path)

    async def main():
        async with TestClient(TestServer(make_app(sup))) as client:
            r = await client.post("/api/setup/devices", json=body)
            assert r.status == 400
            r = await client.post("/api/setup/settings", data=b"not json", headers={"Content-Type": "application/json"})
            assert r.status == 400

    asyncio.run(main())


def test_switch_built_in_artnet_off(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(CONFIG)
    sup, _ = _supervisor(cfg_path)

    async def main():
        pro = FakeTwinkly(pixels=600, api="v2", host="127.0.0.1", artnet=True)
        consumer = FakeTwinkly(pixels=100, api="v1", host="127.0.0.1")
        await pro.start()
        await consumer.start()
        try:
            async with TestClient(TestServer(make_app(sup))) as client:
                r = await client.post("/api/setup/discover", json={"ip": "127.0.0.1", "port": pro.http_port})
                assert (await r.json())["devices"][0]["info"]["artnet_enabled"] is True

                r = await client.post(
                    "/api/setup/artnet", json={"ip": "127.0.0.1", "port": pro.http_port, "enable": False}
                )
                j = await r.json()
                assert r.status == 200, j
                assert j["artnet"] == {"enable": False, "model": 0, "port_address": 0, "dmx_address": 1}
                assert pro.artnet is False and pro.logins == 1
                r = await client.post("/api/setup/discover", json={"ip": "127.0.0.1", "port": pro.http_port})
                assert (await r.json())["devices"][0]["info"]["artnet_enabled"] is False

                r = await client.post(
                    "/api/setup/artnet", json={"ip": "127.0.0.1", "port": pro.http_port, "enable": True}
                )
                assert r.status == 200 and pro.artnet is True

                r = await client.post(
                    "/api/setup/artnet", json={"ip": "127.0.0.1", "port": consumer.http_port, "enable": False}
                )
                assert r.status == 400 and "no built-in Art-Net" in (await r.json())["error"]
                r = await client.post(
                    "/api/setup/artnet", json={"ip": "127.0.0.1", "port": consumer.http_port, "enable": "off"}
                )
                assert r.status == 400
                r = await client.post("/api/setup/artnet", json={"ip": "127.0.0.1", "port": 1, "enable": False})
                assert r.status == 502
        finally:
            await pro.stop()
            await consumer.stop()

    asyncio.run(main())
