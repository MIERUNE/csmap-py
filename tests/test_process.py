import os

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from csmap.process import process, csmap, CsmapParams


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

    assert _csmap.shape == csmap_fixture.shape
    assert _csmap.dtype == csmap_fixture.dtype

    # compare all pixels
    assert (_csmap == csmap_fixture).all()


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

    assert csmap_by_chunk.shape == csmap_fixture.shape
    assert csmap_by_chunk.dtype == csmap_fixture.dtype

    # compare all pixels
    assert (csmap_by_chunk == csmap_fixture).all()


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

    assert csmap_by_worker.shape == csmap_fixture.shape
    assert csmap_by_worker.dtype == csmap_fixture.dtype

    # compare all pixels
    assert (csmap_by_worker == csmap_fixture).all()


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
