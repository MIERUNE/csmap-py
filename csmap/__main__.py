import numpy as np
import rasterio


def calc_slope(dem: np.ndarray) -> np.ndarray:
    """傾斜率を求める
    出力のndarrayのshapeは、(dem.shape[0] - 2, dem.shape[1] - 2)
    """
    z2 = dem[1:-1, 0:-2]
    z4 = dem[0:-2, 1:-1]
    z6 = dem[2:, 1:-1]
    z8 = dem[1:-1, 2:]
    p = (z6 - z4) / 2
    q = (z8 - z2) / 2
    p2 = p * p
    q2 = q * q

    slope = np.arctan((p2 + q2) ** 0.5)
    return slope


def gaussianfilter(image: np.ndarray, size: int, sigma: int) -> np.ndarray:
    """ガウシアンフィルター"""
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = g / g.sum()

    # 画像を畳み込む
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    # パディングサイズを計算
    pad_h, pad_w = k_h // 2, k_w // 2

    # 画像にパディングを適用
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # einsumを使用して畳み込みを行う
    sub_matrices = np.lib.stride_tricks.as_strided(
        padded, shape=(i_h, i_w, k_h, k_w), strides=padded.strides * 2
    )
    return np.einsum("ijkl,kl->ij", sub_matrices, kernel)


def calc_curvature(dem: np.ndarray) -> np.ndarray:
    """曲率を求める"""

    # SAGA の Slope, Aspect, Curvature の 9 parameter 2nd order polynom に準拠
    z1 = dem[0:-2, 0:-2]
    z2 = dem[1:-1, 0:-2]
    z3 = dem[2:, 0:-2]
    z4 = dem[0:-2, 1:-1]
    z5 = dem[1:-1, 1:-1]
    z6 = dem[2:, 1:-1]
    z7 = dem[0:-2, 2:]
    z8 = dem[1:-1, 2:]
    z9 = dem[2:, 2:]

    cell_size = 1
    cell_area = cell_size * cell_size
    r = ((z4 + z6) / 2 - z5) / cell_area
    t = ((z2 + z8) / 2 - z5) / cell_area
    p = (z6 - z4) / (2 * cell_size)
    q = (z8 - z2) / (2 * cell_size)
    s = (z1 - z3 - z7 + z9) / (4 * cell_area)
    p2 = p * p
    q2 = q * q
    spq = s * p * q

    # curvature
    return -2 * (r + t)  # gene

    # plan
    p2_q2 = p2 + q2
    p2_q2 = np.where(p2_q2 > 1e-6, p2_q2, np.nan)
    return -(t * p2 + r * q2 - 2 * spq) / ((p2_q2) ** 1.5)


def rgbify(arr: np.ndarray, method, scale: (float, float) = None) -> np.ndarray:
    """ndarrayをRGBに変換する
    - arrは変更しない
    - ndarrayのshapeは、(4, height, width) 4はRGBA
    """

    _min = arr.min() if scale is None else scale[0]
    _max = arr.max() if scale is None else scale[1]

    # -x ~ x を 0 ~ 1 に正規化
    arr = (arr - _min) / (_max - _min)
    # clamp
    arr = np.where(arr < 0, 0, arr)
    arr = np.where(arr > 1, 1, arr)

    # 3次元に変換
    rgb = method(arr)
    return rgb


def slope_red(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = (1 - arr) * 255  # B
    rgb[3, :, :] = 255
    return rgb


def slope_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = (1 - arr) * 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = (1 - arr) * 255  # B
    rgb[3, :, :] = 255  # A
    return rgb


def curvature_blue(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = (1 - arr) * 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = 255  # B
    rgb[3, :, :] = 255
    return rgb


def curvature_redyellowblue(arr: np.ndarray) -> np.ndarray:
    # value:0-1 to: red -> yellow -> blue
    # interpolate between red and yellow, and yellow and blue, by linear

    # 0-0.5: red -> yellow
    rgb1 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb1[0, :, :] = 255  # R
    rgb1[1, :, :] = arr * 510  # G
    rgb1[2, :, :] = 30 + (arr * 2) * 225  # B

    # 0.5-1: yellow -> blue
    rgb2 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb2[0, :, :] = (1 - (arr * 2 - 1)) * 255  # R
    rgb2[1, :, :] = (1 - (arr * 2 - 1)) * 255  # G
    rgb2[2, :, :] = 30 + (arr * 2 - 1) * 225  # B

    # blend
    rgb = np.where(arr < 0.5, rgb1, rgb2)
    rgb = rgb * 0.5
    rgb[3, :, :] = 255

    return rgb


def height_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = (1 - arr) * 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = (1 - arr) * 255  # B
    rgb[3, :, :] = 255
    return rgb


if __name__ == "__main__":
    dem_path = "./12ke56_1mdem.tif"
    dem = rasterio.open(dem_path).read(1)
    slope = calc_slope(dem)
    g = gaussianfilter(dem, 12, 3)
    curvature = calc_curvature(g)

    dem_rgb = rgbify(dem, height_blackwhite)
    slope_rgb = rgbify(slope, slope_red, scale=(0, 1.5))
    slope_rgb_bw = rgbify(slope, slope_blackwhite, scale=(0, 1.5))
    curvature_rgb_blue = rgbify(curvature, curvature_blue, scale=(-0.1, 0.1))
    curvature_rgb_ryb = rgbify(curvature, curvature_redyellowblue, scale=(-0.1, 0.1))

    dem_rgb = dem_rgb[:, 1:-1, 1:-1]  # remove padding

    # blend all rgb
    blend = np.zeros((4, dem_rgb.shape[0], dem_rgb.shape[1]), dtype=np.uint8)
    blend = (
        dem_rgb * 0.2
        + slope_rgb * 0.15
        + slope_rgb_bw * 0.35
        + curvature_rgb_blue * 0.1
        + curvature_rgb_ryb * 0.20
    )

    # write rgb to tif
    import os

    os.makedirs("output", exist_ok=True)
    profile = rasterio.open(dem_path).profile
    profile.update(dtype=rasterio.uint8, count=4)
    with rasterio.open("output/rgb.tif", "w", **profile) as dst:
        dst.write(blend)
