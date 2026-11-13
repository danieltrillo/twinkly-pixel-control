@echo off
rem Resolume Spout in, Twinkly tree Art-Net out. Edit the map path and the two IPs.
cd /d "%~dp0"
python spout2artnet.py run --map tree.xml --target 1=LUMIVERSE1_IP --target 2=LUMIVERSE2_IP %*
pause
