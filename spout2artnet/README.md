# spout2artnet

Resolume's composition over **Spout** in, the tree's pixels over **Art-Net** out.

Arena's Advanced Output chokes on a whole tree (thousands of fixture slices).
This app takes that job off Arena: Arena only shares its composition over
Spout, and spout2artnet samples the colour under every mapped pixel and sends
the Art-Net itself. It reads the **same Advanced Output XML** the bridge's
`/map` page exports, so the pixel positions, start channels, lumiverses and
Sub-Net/Universe numbers are exactly what Arena would have used.

Runs on the Arena machine (Windows). About 3 ms of CPU per frame for 36,000
pixels.

## Install (once)

1. Python 3.11 or newer from python.org (tick "Add to PATH").
2. In this folder: `pip install -r requirements.txt`

## Arena side

1. Share the composition over Spout: **Output > Spout** (or an Advanced Output
   screen whose device is Spout, slice = whole composition). Keep the
   composition at the size the map was made for (1920x1080); other sizes get
   scaled, so only the aspect ratio really matters.
2. **Turn off the DMX lumiverses in Advanced Output** (or load a preset without
   them). Otherwise Arena and this app both send to the same universes.

`python spout2artnet.py senders` lists the Spout senders it can see.

## Run

```
python spout2artnet.py info --map tree.xml
python spout2artnet.py run  --map tree.xml --target 1=10.0.0.5 --target 2=10.0.0.6
```

`run-tree.bat` is the same thing to double-click (lumiverse 1 to 10.0.0.5, lumiverse 2 to 10.0.0.6).

- `--target N=IP` sends lumiverse N (1, 2, ... in file order, as `info` shows)
  to that IP. Leave it out and the TargetIP saved in the XML is used, if it is
  an IP. `all=IP` sets every lumiverse, `N=off` skips one.
- `--net N=1` sets the Art-Net Net for lumiverse N. Default 0, which is what
  Arena sends (it has no Net field), so the bridge sees the same packets as it
  does from Arena today. Use it if you'd rather send both lumiverses to one IP.
- `--sender NAME` picks a Spout sender (default: the active one).
- `--snapshot check.png` saves what came in 2 s after start, with every sample
  point drawn in magenta. Open it to check the map lines up with the picture.
  Upside down? Add `--flip`.
- `--fps 30` (default, like a lumiverse). `--brightness 80` master level.
- `--gamma` defaults to the fixture Gamma in the XML, or 2.5 (what Arena 7.24.3
  saves for a pixel fixture). `--gamma 1` sends raw values.
- `--white none|min|add` for RGBW pixels. `none` (default) leaves W off, like a
  straight pass-through. `min` moves the grey part of each colour onto the W
  chip. `add` adds W on top (brightest).
- `--area 3` averages a 3x3 patch per pixel instead of the centre pixel
  (smoother on fine detail, about 4x the CPU).
- `--bind IP` sends from a given NIC (the Art-Net one, if the PC has several).
- `--source test` (or `test:rise`, `test:white`) runs a built-in pattern
  instead of Spout, to check the tree without Arena. Works on any OS.
- `--blackout-on-exit` sends black when you stop it. Default is to just stop,
  and the bridge holds the last frame like it does when Arena stops.

Each frame's packets are spaced out (`--gap-us 50`, about 14 ms for a whole
tree) instead of one burst, so the bridge's receive buffer never overflows. The
bridge still does its own per-controller pacing towards the Twinklys.

When Spout stops, the last frame keeps repeating (the status line says
`INPUT LOST, holding`) until Arena comes back.

## Notes

- Pixels are sampled at the centre of each fixture cell. A 1xN line slice runs
  from its v0/v1 end, the way the `/map` export writes it. For WxH grid
  fixtures, pixels run row by row from the v0 corner.
- Colour formats are read per slice (`rgbw`, `rgb`, `grb`, ...).
- NDI input could be added later as another `--source` if Spout ever isn't an
  option.
