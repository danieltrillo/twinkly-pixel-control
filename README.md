# Twinkly Pixel Control

Drive Twinkly Pro controllers **pixel by pixel** from Madrix, Resolume Arena,
grandMA3 or anything else that speaks Art-Net or sACN.

Twinkly's built-in Art-Net mode is not per pixel. It's a 10-channel "fixture"
(dimmer, mode, one colour, movie number, playlist number), so the whole
controller behaves like one lamp. This bridge works around that. It listens
for Art-Net or sACN like a normal pixel node, then streams every pixel to each
controller using Twinkly's own realtime protocol, the same one the Twinkly app
uses for mapping.

```
  Madrix / Arena / MA3                 Twinkly Bridge                  Twinkly Pro
  ┌──────────────────┐   ArtDmx      ┌────────────────┐   realtime     ┌──────────┐
  │ RGBW pixels,     │ ────────────► │ universe map   │ ─────────────► │ Tree 1   │
  │ 128 px/universe  │  0.0.0-0.0.11 │ per-device     │  UDP 7777      │ 1500 px  │
  └──────────────────┘               │ pixel buffers  │ ─────────────► │ Tree 2   │
          ▲            ArtPollReply  │ 25 fps out     │   HTTP 80      │ ...      │
          └───────────────────────── │ status :8080   │  (login, mode) └──────────┘
                                     └────────────────┘
```

It runs on Linux, macOS or a Pi with Python 3.11+. It needs two libraries
(`PyYAML`, `aiohttp`) and no special privileges.

> **Status:** v0.1, **verified on a Twinkly Pro** (TWPROCTRLPLC21, firmware
> 4.0.1, 1440 RGBW pixels on 6 ports) on 2026-11-05: colour order, first and
> last pixel, a walking dot and a rainbow all came out right, both from the
> hardware probe and through the bridge from Art-Net (12 universes at 20 fps,
> no send errors, clean hand-back to the movie it was playing). The realtime
> protocol is unofficial and community documented
> ([xled-docs](https://xled-docs.readthedocs.io/en/latest/protocol_details.html)),
> so step 1 below still matters on other firmware.
>
> **Twinkly Pro firmware 4.x** only speaks `/xled/v2`; the bridge detects that
> and uses the v2 names (`application/mode`, `fw/ct1/version`, nested result
> codes, login before anything else). Two things must be done on every Pro
> controller before a show; see [Before the show](#before-the-show).

## Step 1: prove one controller

Before building a show on it, check that your controller and firmware accept
realtime frames. `tools/twinkly_probe.py` is a single file with no installs
needed:

```bash
python3 tools/twinkly_probe.py info 10.20.30.101
python3 tools/twinkly_probe.py test 10.20.30.101
```

It finds the API generation by itself (consumer `/xled/v1` or Pro firmware
4.x `/xled/v2`) and stops with a clear message if the controller is a group
slave. `test` flashes red, green, blue and then the white chip only. Next it lights
the first pixel, then the last pixel, then sends a walking dot and a rainbow.
After each step it asks what you saw. If colours come out wrong, try
`--order RGBW` (the default is `WRGB`, which is what Twinkly documents for
RGBW). At the end it hands the controller back to whatever it was playing.
Add `--yes --hold 6` to run the whole sequence without prompts, six seconds
per step with a second of black between, which suits one person at the
keyboard and another watching the strands.

## Before the show

On every Twinkly Pro controller:

1. **Take it out of its group** in the Twinkly Pro app. A controller that is
   a slave in a group refuses every output write with HTTP 403, realtime
   included. `twinkly-bridge info <ip>` shows the group state, and the status
   page warns about it.
2. **Turn off the built-in Art-Net/DMX mode** (the 10-channel fixture) so it
   doesn't fight the bridge for the output. `info` shows whether it's on.
3. Know that on firmware 4.0.1 rt mode does not time out by itself. The bridge
   hands each controller back to its previous mode when it stops, but if the
   bridge is killed outright the controller may sit in rt mode showing the
   last frame. Pick a movie in the app to get it back.

## Install

```bash
git clone https://github.com/danieltrillo/twinkly-pixel-control
cd twinkly-pixel-control
sudo ./deploy/install.sh          # Debian / Ubuntu / Raspberry Pi OS: systemd service
```

Or just run it from the checkout:

```bash
pip install pyyaml aiohttp
python3 -m twinklybridge run -c my-config.yaml
```

### In a Proxmox LXC

Use Debian 12, unprivileged, with 1 vCPU, 512 MB RAM and 4 GB disk. That's
plenty for 20+ controllers. Give it 2 vCPU and 4 GB if Claude Code runs in it
too. Put one NIC on the Art-Net/Twinkly VLAN with a static IP. It needs
internet only for `apt` and `git clone`. If the Proxmox firewall is on for
the container, allow UDP 6454 (Art-Net), 5568 (sACN), 5555 (discovery) and
7777 (realtime), plus TCP 8080 (status page). Then run `install.sh` as above.

## Set up 20 controllers in a minute

```bash
twinkly-bridge discover                         # list what's on the network
twinkly-bridge discover --yaml > devices.yaml   # ready-to-paste devices: block
```

`--yaml` asks every controller how many LEDs it has and assigns universes one
after another (`--start-universe 0.1.0` to start somewhere else). Paste the
result into the config, then:

```bash
twinkly-bridge check -c /etc/twinkly-bridge/config.yaml   # prints the universe map
sudo systemctl reload twinkly-bridge                      # apply
```

Open `http://<bridge-ip>:8080` to see every controller: its state, input rate,
output fps, firmware and LED count.

## Config

`/etc/twinkly-bridge/config.yaml`. The annotated version is in
[`twinklybridge/example-config.yaml`](twinklybridge/example-config.yaml), and
`twinkly-bridge example` prints it.

```yaml
input:
  cidr: 2.0.0.20/8        # the bridge's IP on the Art-Net network
  artnet: true
  sacn: false

defaults:
  input: RGBW             # how one pixel is patched on the console
  fps: 25
  on_loss: hold

devices:
  - name: Tree 1
    ip: 10.20.30.101
    universe: 0.0.0       # first universe; 1500 RGBW pixels use 0.0.0 - 0.0.11
    pixels: 1500
  - name: Tree 2
    ip: 10.20.30.102
    universe: 0.0.12
    pixels: 1500
```

| device key | default | meaning |
|---|---|---|
| `ip` | required | The controller's IP. Give it a fixed IP or a DHCP reservation. |
| `universe` | required | First universe. Art-Net is `net.subnet.universe`, sACN is a number. Following universes are used in order. |
| `pixels` | required | Pixel count. The bridge warns if the controller reports a different number. |
| `protocol` | `artnet` | `artnet` or `sacn` for the input. |
| `input` | `RGBW` | Channel layout of one pixel as the console sends it: `RGBW`, `RGB`, `GRBW`, ... |
| `order` | `auto` | Byte order sent to the Twinkly. `auto` uses `WRGB` for RGBW strands and `RGB` for RGB. Override this if colours are swapped. |
| `start_channel` | `1` | Where the first pixel starts in the first universe. |
| `pixels_per_universe` | as many as fit | 128 for RGBW, 170 for RGB. Pixels never straddle universes. |
| `fps` | `25` | Frames per second to the controller. Twinkly renders about 24, so going higher mostly adds traffic. |
| `on_loss` | `hold` | What happens when input stops: `hold` keeps the last look, `black` goes dark, `release` hands back to the app's effect. |
| `loss_timeout` | `5` | Seconds without input before `on_loss` kicks in. |
| `enabled` | `true` | Set to `false` to leave a controller alone. |

Any device key except `name`, `ip`, `universe`, `pixels` and `enabled` can go
under `defaults:`.

## Patch it

**Madrix / Arena.** Create RGBW pixel fixtures at 128 pixels per universe
(170 for RGB), starting at the device's first universe. Point the Art-Net
output at the bridge's IP (unicast). The bridge also answers ArtPoll, so it
shows up as **TwinklyBridge** in node lists.

**grandMA3.** Patch a 4-channel RGBW pixel fixture, 128 per universe. Add
Art-Net output lines for the universes and set the destination to the bridge
IP. Then use layouts and the Bitmap/pixel mapper as usual.

**Turn off Twinkly's own Art-Net mode** on the controllers (Twinkly app →
Device → Advanced). If it's on and listening to the same universes, it fights
the bridge.

## What the bridge does with each controller

1. Logs in over HTTP and reads the model, firmware, LED count and RGB/RGBW.
   Until input arrives it leaves the controller alone, so whatever the app
   was playing keeps running.
2. When the first universe arrives, it switches the controller to realtime
   mode and starts streaming at `fps`.
3. Every 5 s it checks the controller is still in realtime mode, and puts it
   back if someone pressed a button in the app.
4. It re-sends the last frame every 0.5 s even when nothing changes, because
   realtime mode times out without frames.
5. If the controller drops off the network, it retries with backoff and picks
   up where it left off. It logs in again when the token expires (about every
   4 hours).
6. On shutdown it puts each controller back in the mode it was in before.

## Testing without gear

```bash
twinkly-bridge fake --pixels 300 --http-port 8081 --rt-port 7778   # a pretend controller
twinkly-bridge send --to 127.0.0.1 --universe 0.0.0-0.0.2 --pattern rainbow
```

For a device, set `ip: 127.0.0.1`, `http_port: 8081` and `rt_port: 7778` to
point the bridge at the fake. It prints the fps it receives and the first
pixel's bytes. `--api v2` makes it behave like Pro firmware 4.x, and
`--group-slave` makes it refuse output writes the way a grouped Pro
controller does.

## Commands

| command | what it does |
|---|---|
| `run -c FILE` | Run the bridge (default command). |
| `check -c FILE` | Validate the config and print the universe map. |
| `discover [--yaml]` | Find controllers (UDP 5555 broadcast); `--yaml` writes the devices block. |
| `info IP` | Show what one controller reports: firmware, LEDs, outputs, current mode. |
| `send` | Send an Art-Net test pattern (rainbow, chase, solid, white). |
| `fake` | Run a pretend controller. |
| `example` | Print the annotated example config. |

## Troubleshooting

| symptom | check |
|---|---|
| Device shows **offline** | Can the bridge reach it? `twinkly-bridge info <ip>`. Is it on the same VLAN, or is there a route? |
| **ready, waiting for input** | No Art-Net is arriving for its universes. Check the "not mapped" list on the status page, which usually means a universe numbering mix-up (0-based vs 1-based). |
| Colours swapped | Set `order:` on the device (try `RGBW`, `GRBW`). Or check that `input:` matches how the console patches the pixel. |
| Flicker or two looks fighting | Twinkly's own Art-Net mode is on, or the Twinkly app is open and mapping. |
| Low fps out | Look at "send errors" on the status page. Twinkly renders about 24 fps, so above that frames are dropped by the controller. |
| Pixel count warning | The controller reports a different LED count than `pixels:`. The bridge sends what the controller reports. |
| **set mode 'rt' refused (403)** or a "group slave" warning | Pro firmware: the controller is a slave in a group and refuses all output writes. Take it out of its group in the Twinkly Pro app. |
| Controller stuck on the last frame after the bridge died | It's still in rt mode (Pro firmware doesn't time out). Pick a movie in the app, or start the bridge again and stop it cleanly. |

## Not in this version

- ArtSync is ignored. Frames go out on the bridge's own clock at `fps`.
- No merging. If two sources send the same universe, the last packet wins.
- The config is edited as a file (`reload` applies it). The status page is
  read-only.

## Development

```bash
pip install pyyaml aiohttp pytest ruff
pytest -q
ruff check twinklybridge tests tools
```

Layout: `twinkly.py` is the Twinkly HTTP + UDP protocol, `engine.py` handles
input, universe mapping and per-device streaming, `config.py` the YAML
schema, `pixels.py` the channel reordering, `testtools.py` the fake
controller, and `tools/twinkly_probe.py` the standalone hardware test. The
Art-Net and sACN codecs come from [ArtBridge](https://github.com/danieltrillo/artnet-bridge).

## Credits

Twinkly protocol notes: [xled-docs](https://xled-docs.readthedocs.io).
Twinkly is a trademark of LEDWORKS. This project is not affiliated with them.
Art-Net™ Designed by and Copyright Artistic Licence Holdings Ltd. MIT
licensed, see [LICENSE](LICENSE).
