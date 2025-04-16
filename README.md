# CityCAT on DAFNI

[![build](https://github.com/OpenCLIM/citycat-dafni/workflows/build/badge.svg)](https://github.com/OpenCLIM/citycat-dafni/actions)

This repo contains the files required to build and test the citycat-dafni model.
The binary executable for CityCAT is encrypted as this software is not publicly available.
[`Wine`](https://www.winehq.org/) is used to run the executable as it was built for Windows but is required to run on Linux.
All processing steps are contained in [`run.py`](https://github.com/OpenCLIM/citycat-dafni/blob/master/run.py).

## Documentation
[citycat-dafni.md](https://github.com/OpenCLIM/citycat-dafni/blob/master/docs/citycat-dafni.md)

To build the documentation:
```
cd docs
python build_docs.py
```

## Dependencies
[environment.yml](https://github.com/OpenCLIM/citycat-dafni/blob/master/environment.yml)

## Usage 
```
docker build -t citycat_dafni .
docker run -e DURATION=1 -e TOTAL_DEPTH=40 -e OPEN_BOUNDARIES=True -e ROOF_STORAGE=0 -e POST_EVENT_DURATION=0 -e OUTPUT_INTERVAL=3600 -e RAINFALL_MODE=total_depth -e DISCHARGE=20  --name citycat_dafni citycat_dafni
```

or

```
setenv.bat
python run.py
```
