# CityCAT on DAFNI

## Features
- Extract domain from OS Terrain 5
- Generate return period events from UKCP18
- Create archive of results

## Usage 
`docker build -t citycat-dafni . && docker run -v "data:/data" --env PYTHONUNBUFFERED=1 --env SIZE=0.1 --env DURATION=1 --env RETURN_PERIOD=100 --env X=258722 --env Y=665028 --env POOLING_RADIUS=20 --name citycat-dafni citycat-dafni `
