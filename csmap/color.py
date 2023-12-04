import numpy as np


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


def blend(
    dem_bw: np.ndarray,
    slope_red: np.ndarray,
    slope_bw: np.ndarray,
    curvature_blue: np.ndarray,
    curvature_ryb: np.ndarray,
    blend_params: dict = {
        "dem": 0.2,
        "slope_red": 0.15,
        "slope_bw": 0.35,
        "curvature_blue": 0.1,
        "curvature_ryb": 0.2,
    },
) -> np.ndarray:
    """blend all rgb
    全てのndarrayは同じshapeであること
    DEMを用いて処理した他の要素は、DEMよりも1px内側にpaddingされているので
    あらかじめDEMのpaddingを除外しておく必要がある
    """
    _blend = np.zeros((4, dem_bw.shape[0], dem_bw.shape[1]), dtype=np.uint8)
    _blend = (
        dem_bw * blend_params["dem"]
        + slope_red * blend_params["slope_red"]
        + slope_bw * blend_params["slope_bw"]
        + curvature_blue * blend_params["curvature_blue"]
        + curvature_ryb * blend_params["curvature_ryb"]
    )
    return _blend
