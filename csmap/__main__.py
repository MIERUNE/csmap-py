from csmap import calc
from csmap import color

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import numpy as np


# TODO: 各種パラメータを引数で受け取る
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


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dem_path", type=str)
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()
    return {
        "dem_path": args.dem_path,
        "chunk_size": args.chunk_size,
    }


if __name__ == "__main__":
    args = parse_args()

    with rasterio.open(args["dem_path"]) as dem:
        chunk_size = args["chunk_size"]
        margin = 15  # ガウシアンフィルタのサイズ+シグマ
        # チャンクごとの処理結果には「淵=margin」が生じるのでこの部分を除外する必要がある
        margin_to_removed = 2 * (margin // 2)  # 整数値に切り捨てた値*両端

        # write rgb to tif
        import os

        os.makedirs("output", exist_ok=True)

        # マージンを考慮したtransform
        transform = Affine(
            dem.transform.a,
            dem.transform.b,
            dem.transform.c + (1 + margin // 2) * dem.transform.a,  # 左端の座標をマージン分ずらす
            dem.transform.d,
            dem.transform.e,
            dem.transform.f + (1 + margin // 2) * dem.transform.e,  # 上端の座標をマージン分ずらす
            0.0,
            0.0,
            1.0,
        )

        # 生成されるCS立体図のサイズ
        out_width = dem.shape[1] - margin_to_removed - 2
        out_height = dem.shape[0] - margin_to_removed - 2

        with rasterio.open(
            "output/rgb.tif",
            "w",
            driver="GTiff",
            dtype=rasterio.uint8,
            count=4,
            width=out_width,
            height=out_height,
            crs=dem.crs,
            transform=transform,
        ) as dst:
            # chunkごとに処理
            chunk_csmap_size = chunk_size - margin_to_removed - 2
            for y in range(0, dem.shape[0], chunk_csmap_size):
                for x in range(0, dem.shape[1], chunk_csmap_size):
                    chunk = dem.read(1, window=Window(x, y, chunk_size, chunk_size))
                    csmap_chunk = csmap(
                        chunk
                    )  # shape = (4, chunk_size-2, chunk_size-2)
                    csmap_chunk_margin_removed = csmap_chunk[
                        :,
                        margin // 2 : -(margin // 2),
                        margin // 2 : -(margin // 2),
                    ]  # shape = (4, chunk_csmap_size, chunk_csmap_size)

                    # csmpのどの部分を出力用配列に入れるかを計算
                    write_size_x = chunk_csmap_size
                    write_size_y = chunk_csmap_size
                    if x + chunk_csmap_size > out_width:
                        write_size_x = out_width - x
                    if y + chunk_csmap_size > out_height:
                        write_size_y = out_height - y

                    dst.write(
                        csmap_chunk_margin_removed,
                        window=Window(x, y, write_size_x, write_size_y),
                    )
