# CityCAT on DAFNI

[![build](https://github.com/OpenCLIM/citycat-dafni/workflows/build/badge.svg)](https://github.com/OpenCLIM/citycat-dafni/actions)

## Features
- Extract domain from OS Terrain 5
- Generate return period events from UKCP18
- Create archive of results
- Create GeoTIFF of max depth
- Create PNG map of max depth

## Usage 
`docker build -t citycat-dafni . && docker run -v "data:/data" --env PYTHONUNBUFFERED=1 --env RAINFALL_MODE=return_period --env TOTAL_DEPTH=40 --env SIZE=0.1 --env DURATION=1 --env POST_EVENT_DURATION=0 --env RETURN_PERIOD=100 --env X=258722 --env Y=665028 --env POOLING_RADIUS=20 --name citycat-dafni citycat-dafni `
