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
`docker build -t citycat-dafni . && docker run -v $PWD/data:/data --env PYTHONUNBUFFERED=1 --env RAINFALL_MODE=return_period --env SIZE=0.1 --env DURATION=1 --env POST_EVENT_DURATION=0 --env TOTAL_DEPTH=40 --env RETURN_PERIOD=100 --env X=258722 --env Y=665028 --env OPEN_BOUNDARIES=True --env PERMEABLE_AREAS=polygons --env ROOF_STORAGE=0 --env TIME_HORIZON=2050 --name citycat-dafni-return-period citycat-dafni`

or

`python run.py`