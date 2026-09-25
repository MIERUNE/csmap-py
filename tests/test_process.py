import os

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from csmap.process import process, csmap, CsmapParams


def assert_almost_equal(actual: np.ndarray, expected: np.ndarray):
    """浮動小数点演算の結果はCPU(SIMD命令の有無など)によってわずかに異なり、
    uint8への切り捨てで±1の差が生じることがあるため、その範囲の差は許容する"""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 1


def test_csmap():
    """リグレッションがないか確認する"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")
    dem = rasterio.open(dem_path).read(1)

    _csmap = csmap(
        dem,
        params=CsmapParams(
            gf_size=12,
            gf_sigma=3,
            curvature_size=1,
            height_scale=[0, 1000],
            slope_scale=[0, 1.5],
            curvature_scale=[-0.1, 0.1],
        ),
    )

    csmap_fixture_path = os.path.join(os.path.dirname(__file__), "fixture", "csmap.tif")
    csmap_fixture = rasterio.open(csmap_fixture_path).read([1, 2, 3, 4])

    assert_almost_equal(_csmap, csmap_fixture)


def test_process_by_chunk():
    """チャンクごとに処理した結果と一度に処理した結果が一致することをテスト"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")

    csmap_params = CsmapParams(
        gf_size=12,
        gf_sigma=3,
        curvature_size=1,
        height_scale=[0, 1000],
        slope_scale=[0, 1.5],
        curvature_scale=[-0.1, 0.1],
    )

    csmap_by_chunk_path = os.path.join(os.path.dirname(__file__), "test_chunk.tif")
    process(
        input_dem_path=dem_path,
        output_path=csmap_by_chunk_path,
        chunk_size=256,
        params=csmap_params,
        max_workers=2,
    )
    csmap_by_chunk = rasterio.open(csmap_by_chunk_path).read([1, 2, 3, 4])

    # チャンク分割なしに処理した結果と比較
    csmap_fixture_path = os.path.join(
        os.path.dirname(__file__), "fixture", "process.tif"
    )
    csmap_fixture = rasterio.open(csmap_fixture_path).read([1, 2, 3, 4])

    assert_almost_equal(csmap_by_chunk, csmap_fixture)


def test_process_by_worker():
    """並列処理をしても結果に影響がないことをテスト"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")

    csmap_params = CsmapParams(
        gf_size=12,
        gf_sigma=3,
        curvature_size=1,
        height_scale=[0, 1000],
        slope_scale=[0, 1.5],
        curvature_scale=[-0.1, 0.1],
    )

    csmap_by_worker_path = os.path.join(os.path.dirname(__file__), "test_worker.tif")
    process(
        input_dem_path=dem_path,
        output_path=csmap_by_worker_path,
        chunk_size=1024,
        params=csmap_params,
        max_workers=2,
    )
    csmap_by_worker = rasterio.open(csmap_by_worker_path).read([1, 2, 3, 4])

    # チャンク分割なしに処理した結果と比較
    csmap_fixture_path = os.path.join(
        os.path.dirname(__file__), "fixture", "process.tif"
    )
    csmap_fixture = rasterio.open(csmap_fixture_path).read([1, 2, 3, 4])

    assert_almost_equal(csmap_by_worker, csmap_fixture)


def test_process_nodata_transparent(tmp_path):
    """入力DEMのNoData範囲が出力で透過になることをテスト"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")
    with rasterio.open(dem_path) as src:
        profile = src.profile
        dem = src.read(1)

    nodata = -9999.0
    dem[500:700, 800:1200] = nodata
    nodata_dem_path = tmp_path / "dem_nodata.tif"
    profile.update(nodata=nodata)
    with rasterio.open(nodata_dem_path, "w", **profile) as dst:
        dst.write(dem, 1)

    csmap_params = CsmapParams()
    offset = 1 + (csmap_params.gf_size + csmap_params.gf_sigma) // 2

    results = []
    for chunk_size, max_workers in [(4096, 1), (256, 2)]:
        output_path = tmp_path / f"csmap_{chunk_size}.tif"
        process(
            input_dem_path=str(nodata_dem_path),
            output_path=str(output_path),
            chunk_size=chunk_size,
            params=csmap_params,
            max_workers=max_workers,
        )
        with rasterio.open(output_path) as out:
            result = out.read([1, 2, 3, 4])
            assert out.colorinterp[3] == ColorInterp.alpha
        results.append(result)

        # 出力画素に対応する入力DEMのNoData範囲
        expected_mask = (dem == nodata)[
            offset : offset + result.shape[1], offset : offset + result.shape[2]
        ]
        assert (result[3][expected_mask] == 0).all()
        assert (result[3][~expected_mask] == 255).all()

    # チャンク分割・並列処理の有無で結果が一致すること
    assert (results[0] == results[1]).all()


def test_csmap_nan_transparent():
    """NaNの画素が透過になることをテスト"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")
    dem = rasterio.open(dem_path).read(1)[:300, :300]
    dem[100:150, 100:150] = np.nan

    _csmap = csmap(dem, CsmapParams())

    expected_mask = np.isnan(dem)[1:-1, 1:-1]
    assert (_csmap[3][expected_mask] == 0).all()
    assert (_csmap[3][~expected_mask] == 255).all()


def test_csmap_integer_dem():
    """整数型のDEMでも、同じ値のfloat32のDEMと結果が一致することをテスト
    (uint16などで差分計算がオーバーフローしないこと)"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")
    dem = rasterio.open(dem_path).read(1)[:300, :300]
    dem = np.clip(np.round(dem), 0, 3000)

    expected = csmap(dem.astype(np.float32), CsmapParams())
    for dtype in ["int16", "uint16", "int32"]:
        _csmap = csmap(dem.astype(dtype), CsmapParams())
        assert (_csmap == expected).all(), dtype


def _is_bigtiff(path) -> bool:
    """TIFFヘッダのバージョン番号でBigTIFF(43)かClassic TIFF(42)かを判定する"""
    with open(path, "rb") as f:
        header = f.read(4)
    byteorder = "little" if header[:2] == b"II" else "big"
    return int.from_bytes(header[2:4], byteorder) == 43


def test_process_bigtiff(tmp_path):
    """bigtiffオプションが出力に反映されることをテスト"""
    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")

    for bigtiff, expected in [("YES", True), ("NO", False), ("IF_SAFER", False)]:
        output_path = tmp_path / f"csmap_{bigtiff}.tif"
        process(
            input_dem_path=dem_path,
            output_path=str(output_path),
            chunk_size=1024,
            params=CsmapParams(),
            bigtiff=bigtiff,
        )
        assert _is_bigtiff(output_path) == expected, bigtiff

    # BigTIFFでも内容は変わらないこと
    with rasterio.open(tmp_path / "csmap_YES.tif") as a, rasterio.open(
        tmp_path / "csmap_NO.tif"
    ) as b:
        assert (a.read() == b.read()).all()


def test_process_raises_in_worker(tmp_path, monkeypatch):
    """並列処理時にスレッド内の例外が握りつぶされず送出されることをテスト"""
    import pytest

    from csmap import process as process_module

    def _raise(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(process_module, "_process_chunk", _raise)

    dem_path = os.path.join(os.path.dirname(__file__), "fixture", "dem.tif")
    with pytest.raises(RuntimeError, match="write failed"):
        process(
            input_dem_path=dem_path,
            output_path=str(tmp_path / "csmap.tif"),
            chunk_size=256,
            params=CsmapParams(),
            max_workers=2,
        )
