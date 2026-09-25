import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csmap.process import process, CsmapParams


PARAMS = CsmapParams(
    gf_size=12,
    gf_sigma=3,
    curvature_size=1,
    height_scale=[0, 1000],
    slope_scale=[0, 1.5],
    curvature_scale=[-0.1, 0.1],
)


def _make_dem(path: str, dtype: str, nodata, size: int = 600):
    """中央に山、周囲がNoData の DEM を生成する

    Returns:
        NoData 画素の割合
    """
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    elev = 300 * np.exp(-((r / (size * 0.18)) ** 2)) + 20 * np.sin(xx / 9) * np.exp(
        -((r / (size * 0.22)) ** 2)
    )

    hole = r > size * 0.36  # 外周をNoDataにする
    if np.issubdtype(np.dtype(dtype), np.integer):
        arr = np.clip(elev, 0, 3000).astype(dtype)
    else:
        arr = elev.astype(dtype)
    arr[hole] = nodata

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=dtype,
        nodata=nodata,
        transform=from_origin(139.0, 35.0, 0.5, 0.5),
        crs="EPSG:3857",
    ) as dst:
        dst.write(arr, 1)

    return hole.mean()


@pytest.mark.parametrize(
    "dtype,nodata",
    [
        ("float32", -9999.0),
        ("float32", float("nan")),
        ("float32", 1.70141e38),  # 航空レーザ測量DEMで使われる値
        ("int16", -9999),
        ("int32", -32768),
        ("uint16", 65535),
    ],
)
def test_nodata_is_transparent(tmp_path, dtype, nodata):
    """NoData の範囲が透明になることをテスト

    dtype と NoData 値の組み合わせによらず透明化されること。
    整数型の DEM は NaN を保持できないため、内部で float に昇格する必要がある。
    """
    dem_path = str(tmp_path / f"dem_{dtype}.tif")
    nodata_ratio = _make_dem(dem_path, dtype, nodata)

    out_path = str(tmp_path / f"csmap_{dtype}.tif")
    process(
        input_dem_path=dem_path,
        output_path=out_path,
        chunk_size=256,
        params=PARAMS,
        max_workers=1,
    )

    alpha = rasterio.open(out_path).read(4)
    transparent_ratio = (alpha == 0).mean()

    # 縁が削られる分だけ NoData 比率とはわずかにずれるため許容幅を持たせる
    assert transparent_ratio == pytest.approx(nodata_ratio, abs=0.05), (
        f"{dtype}/{nodata}: 透明={transparent_ratio:.3f} "
        f"期待={nodata_ratio:.3f}"
    )


def test_nodata_transparent_by_worker(tmp_path):
    """並列処理でも NoData の透明化が一致することをテスト"""
    dem_path = str(tmp_path / "dem.tif")
    _make_dem(dem_path, "float32", -9999.0)

    outs = []
    for workers in (1, 2):
        out_path = str(tmp_path / f"csmap_w{workers}.tif")
        process(
            input_dem_path=dem_path,
            output_path=out_path,
            chunk_size=256,
            params=PARAMS,
            max_workers=workers,
        )
        outs.append(rasterio.open(out_path).read([1, 2, 3, 4]))

    assert outs[0].shape == outs[1].shape
    assert (outs[0] == outs[1]).all()


def test_no_nodata_declared_stays_opaque(tmp_path):
    """NoData が宣言されていない DEM では全て不透明のままであることをテスト

    透明化は NoData 宣言を根拠にしているため、宣言が無ければ何も起きない。
    既存の挙動が変わらないことの確認でもある。
    """
    dem_path = str(tmp_path / "dem_no_nodata.tif")
    _make_dem(dem_path, "float32", -9999.0)

    # NoData 宣言だけを外す（画素値は変えない）
    with rasterio.open(dem_path, "r+") as dst:
        dst.nodata = None

    out_path = str(tmp_path / "csmap_no_nodata.tif")
    process(
        input_dem_path=dem_path,
        output_path=out_path,
        chunk_size=256,
        params=PARAMS,
        max_workers=1,
    )

    alpha = rasterio.open(out_path).read(4)
    assert (alpha == 255).all()
