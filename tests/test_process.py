import os

import rasterio

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
