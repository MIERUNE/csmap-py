from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

from csmap import calc
from csmap import color


@dataclass
class CsmapParams:
    gf_size: int
    gf_sigma: int
    cvt_size: int
    height_scale: (float, float)
    slope_scale: (float, float)
    curvature_scale: (float, float)


def csmap(dem: np.ndarray, params: CsmapParams) -> np.ndarray:
    """DEMからCS立体図を作成する"""
    # calclucate elements
    slope = calc.slope(dem)
    g = calc.gaussianfilter(dem, params.gf_size, params.gf_sigma)
    curvature = calc.curvature(g, params.cvt_size)

    # rgbify
    dem_rgb = color.rgbify(dem, color.height_blackwhite, scale=params.height_scale)
    slope_red = color.rgbify(slope, color.slope_red, scale=params.slope_scale)
    slope_bw = color.rgbify(slope, color.slope_blackwhite, scale=params.slope_scale)
    curvature_blue = color.rgbify(
        curvature, color.curvature_blue, scale=params.curvature_scale
    )
    curvature_ryb = color.rgbify(
        curvature, color.curvature_redyellowblue, scale=params.curvature_scale
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


def process(
    input_dem_path: str, output_path: str, chunk_size: int, params: CsmapParams
):
    with rasterio.open(input_dem_path) as dem:
        margin = params.gf_size + params.gf_sigma  # ガウシアンフィルタのサイズ+シグマ
        # チャンクごとの処理結果には「淵=margin」が生じるのでこの部分を除外する必要がある
        margin_to_removed = 2 * (margin // 2)  # 整数値に切り捨てた値*両端

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
            output_path,
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
                        chunk, params
                    )  # shape=(4,chunk_size-2,chunk_size-2)
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
