@echo off
rem Resolume Spout in, Twinkly tree Art-Net out. Edit the map path if needed. Lumiverse 1 -> 10.0.0.5, Lumiverse 2 -> 10.0.0.6.
cd /d "%~dp0"
python spout2artnet.py run --map tree.xml --target 1=10.0.0.5 --target 2=10.0.0.6 %*
pause
