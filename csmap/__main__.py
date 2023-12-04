import calc
import color

import rasterio
import numpy as np


def csmap(dem: np.ndarray) -> np.ndarray:
    """DEMからCS立体図を作成する"""
    # calclucate elements
    slope = calc.slope(dem)
    g = calc.gaussianfilter(dem, 12, 3)
    curvature = calc.curvature(g, 1)

    # rgbify
    dem_rgb = color.rgbify(dem, color.height_blackwhite, scale=(0, 1000))
    slope_red = color.rgbify(slope, color.slope_red, scale=(0, 1.5))
    slope_bw = color.rgbify(slope, color.slope_blackwhite, scale=(0, 1.5))
    curvature_blue = color.rgbify(curvature, color.curvature_blue, scale=(-0.1, 0.1))
    curvature_ryb = color.rgbify(
        curvature, color.curvature_redyellowblue, scale=(-0.1, 0.1)
    )

    dem_rgb = dem_rgb[:, 1:-1, 1:-1]  # remove padding

    # blend all rgb
    blend = color.blend(
        dem_rgb,
        slope_red,
        slope_bw,
        curvature_blue,
        curvature_ryb,
    )

    return blend


if __name__ == "__main__":
    dem_path = "./12ke47_1mdem.tif"
    dem = rasterio.open(dem_path).read(1)

    chunk_size = 1024
    margin = 15  # ガウシアンフィルタのサイズ+シグマ：チャンクごとに「淵」が生じるのでこの部分は除外する必要がある
    _csmap = np.zeros(
        (4, dem.shape[0] - 2, dem.shape[1] - 2), dtype=np.uint8
    )  # TODO: ハードコードをやめる

    # chunkごとに処理
    for y in range(0, dem.shape[0] - 2, chunk_size - margin):
        for x in range(0, dem.shape[1] - 2, chunk_size - margin):
            print(f"y={y}, x={x}")
            chunk = dem[y : y + chunk_size, x : x + chunk_size]
            csmap_chunk = csmap(chunk)
            _csmap[
                :,
                y + margin // 2 : y + chunk_size - margin // 2,
                x + margin // 2 : x + chunk_size - margin // 2,
            ] = csmap_chunk[
                :,
                margin // 2 : chunk_size - margin // 2,
                margin // 2 : chunk_size - margin // 2,
            ]

    # write rgb to tif
    import os

    os.makedirs("output", exist_ok=True)

    # TODO: TIFの領域はオリジナルのDEMより少し狭い（1pxのパディングがある）
    profile = rasterio.open(dem_path).profile
    profile.update(dtype=rasterio.uint8, count=4)
    with rasterio.open("output/rgb.tif", "w", **profile) as dst:
        dst.write(_csmap)
