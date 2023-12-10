# csmap-py

![](./doc/csmap.png)

module to process CSMap, based on https://www.rinya.maff.go.jp/j/seibi/sagyoudo/attach/pdf/romou-12.pdf

```
usage: __main__.py [-h] [--chunk_size CHUNK_SIZE] [--max_workers MAX_WORKERS] [--gf_size GF_SIZE]
                   [--gf_sigma GF_SIGMA] [--curvature_size CURVATURE_SIZE]
                   [--height_scale HEIGHT_SCALE HEIGHT_SCALE]
                   [--slope_scale SLOPE_SCALE SLOPE_SCALE]
                   [--curvature_scale CURVATURE_SCALE CURVATURE_SCALE]
                   input_dem_path output_path

positional arguments:
  input_dem_path
  output_path

options:
  -h, --help            show this help message and exit
  --chunk_size CHUNK_SIZE
  --max_workers MAX_WORKERS
  --gf_size GF_SIZE
  --gf_sigma GF_SIGMA
  --curvature_size CURVATURE_SIZE
  --height_scale HEIGHT_SCALE HEIGHT_SCALE
  --slope_scale SLOPE_SCALE SLOPE_SCALE
  --curvature_scale CURVATURE_SCALE CURVATURE_SCALE
```

```sh
poetry run python -m csmap 12ke47_1mdem.tif --chunk_size 2048 --height_scale 200 500
```

## processing image

![](./doc/process.jpeg)