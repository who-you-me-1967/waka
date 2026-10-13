import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares, minimize_scalar

g_path_img_ellipse = "C:\\py\\data\\img2\\test_ellipse.png"
g_path_img_ellipse_0 = "C:\\py\\data\\img2\\test_ellipse_0.png"
g_path_img_ellipse_1 = "C:\\py\\data\\img2\\test_ellipse_1.png"
g_path_img_ellipse_out = "C:\\py\\data\\img2\\test_ellipse_out.png"


def rotate_vector(v, va, rad):
    """
    ベクトル v を、軸 va の周りに rad ラジアン回転させる。

    Parameters
    ----------
    v : array-like, shape (3,)
        回転させるベクトル
    va : array-like, shape (3,)
        回転軸ベクトル
    rad : float
        回転角度（ラジアン、右手の法則）

    Returns
    -------
    numpy.ndarray, shape (3,)
        回転後のベクトル
    """
    v = np.asarray(v, dtype=float)
    va = np.asarray(va, dtype=float)

    norm = np.linalg.norm(va)
    if norm == 0:
        raise ValueError("回転軸 va はゼロベクトルにできません")

    # 回転軸を正規化
    k = va / norm

    # ロドリゲスの回転公式
    return v * np.cos(rad) + np.cross(k, v) * np.sin(rad) + k * np.dot(k, v) * (1.0 - np.cos(rad))


def fit_ellipse_least_squares(binary_img, binary_img2):
    """
    二値画像 (値: 0 or 1, または 0 or 255) から最小二乗法を用いて楕円を検出する。

    Parameters:
        binary_img (numpy.ndarray): 入力の二値化画像

    Returns:
        dict or None: 検出された楕円パラメータ
            - 'center': (cx, cy) 中心座標
            - 'axes': (a, b) 長軸および短軸の半径
            - 'angle': rad 楕円の回転角（ラジアン）
    """
    y_indices, x_indices = np.where(0 < binary_img)
    y_indices2, x_indices2 = np.where(0 < binary_img2)
    # print(y_indices)
    # exit()

    # 楕円を特定するには最低5点以上のデータが必要
    if len(x_indices) < 5 or len(x_indices2) < 5:
        print("エラー: 点の数が不足しています (最低5点必要)")
        return None

    x = x_indices.astype(np.float64)
    y = y_indices.astype(np.float64)
    x2 = x_indices2.astype(np.float64)
    y2 = y_indices2.astype(np.float64)

    # 2. 残差関数の定義
    # 代数方程式: A*x^2 + B*x*y + C*y^2 + D*x + E*y + 1 = 0 の残差
    def residual(params, x, y, x2, y2):
        A, B, C, D, E = params
        return A * x**2 + B * x * y + C * y**2 + D * x + E * y + 1.0

    # 3. 初期値の設定と scipy.optimize.least_squares による最適化
    # 円に近い形状を仮定した初期値
    initial_params = [1.0, 0.0, 1.0, 0.0, 0.0]
    result = least_squares(residual, initial_params, args=(x, y, x2, y2))

    A, B, C, D, E = result.x

    # 4. 代数係数 (A, B, C, D, E, 1) から幾何学パラメータへの変換
    # 判別式 (B^2 - 4AC < 0 で楕円)
    det = B**2 - 4 * A * C
    if det >= 0:
        print("警告: 最小二乗法の解が楕円条件を満たしていません")
        return None

    # 中心座標 (cx, cy)
    cx = (2 * C * D - B * E) / det
    cy = (2 * A * E - B * D) / det

    # 軸長計算用の中間値
    term = 2 * (A * E**2 + C * D**2 - B * D * E + det)
    scale1 = np.sqrt(1 + (A - C) ** 2 / B**2) if B != 0 else 1.0

    # 半径 (a, b)
    a = np.sqrt(np.abs(term / (det * (np.sqrt((A - C) ** 2 + B**2) - (A + C)))))
    b = np.sqrt(np.abs(term / (det * (-np.sqrt((A - C) ** 2 + B**2) - (A + C)))))

    # 回転角度（ラジアン）
    if B == 0:
        angle_rad = 0.0 if A < C else np.pi / 2
    else:
        angle_rad = 0.5 * (np.arctan2(B, A - C) + np.pi)

    return (float(cx), float(cy), float(a), float(b), float(angle_rad))


def get_parameters_of_ellipse(x, y, z, ax, ay, az, r):
    """
    原点にあるカメラから、3D円を z=-1 平面へ透視投影する。

    Parameters
    ----------
    x, y, z : float
        3D円の中心座標
    ax, ay, az : float
        円の平面の法線方向（ゼロベクトル以外）
    r : float
        円の半径

    Returns
    -------
    x, y : float
        投影された楕円の中心座標
    a, b : float
        楕円の長半径、短半径（a >= b）
    angle_rad : float
        長軸とX軸の正方向との角度（ラジアン）

    Raises
    ------
    ValueError
        投影が退化する場合、または実楕円にならない場合
    """

    if r <= 0:
        raise ValueError("r must be positive")

    C = np.array([x, y, z], dtype=float)
    n = np.array([ax, ay, az], dtype=float)

    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("The normal vector must be non-zero")

    n /= norm

    # 円がカメラの視点を通る平面上にある場合は投影が退化する
    if abs(np.dot(n, C)) < 1e-12:
        raise ValueError("Degenerate projection: circle plane passes through camera")

    # カメラ側（z < 0）に円全体が存在することを確認
    z_extent = r * np.sqrt(max(0.0, 1.0 - n[2] ** 2))
    if z + z_extent >= 0:
        raise ValueError("The entire circle must lie in front of the camera (z < 0)")

    # 円の平面内の直交基底 u, v を構築
    if abs(n[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])

    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # 3D平面上の座標 (s, t) から投影画像への射影変換 H
    # 画像座標は (X / -Z, Y / -Z)
    H = np.array([[u[0], v[0], C[0]], [u[1], v[1], C[1]], [-u[2], -v[2], -C[2]]], dtype=float)

    # 円の方程式 s^2 + t^2 - r^2 = 0
    Q_circle = np.diag([1.0, 1.0, -r * r])

    # 画像上の円錐曲線の行列
    H_inv = np.linalg.inv(H)
    Q = H_inv.T @ Q_circle @ H_inv

    # 対称行列に整える（数値誤差対策）
    Q = (Q + Q.T) / 2.0

    # Ax^2 + 2Bxy + Cy^2 + 2Dx + 2Ey + F = 0
    A2 = Q[:2, :2]
    d = Q[:2, 2]
    F = Q[2, 2]

    # 楕円の中心
    center = -np.linalg.solve(A2, d)

    # 中心移動後の方程式:
    # (p-center)^T A2 (p-center) = k
    k = float(d @ np.linalg.solve(A2, d) - F)

    # A2 と k の符号をそろえる
    eigvals, eigvecs = np.linalg.eigh(A2)

    if np.all(eigvals < 0):
        eigvals = -eigvals
        k = -k
    elif not np.all(eigvals > 0):
        raise ValueError("The projection is not a bounded ellipse")

    if k <= 0:
        raise ValueError("The projection does not form a real ellipse")

    # 固有値の小さい方向が長軸
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = np.sqrt(k / eigvals[0])
    b = np.sqrt(k / eigvals[1])

    # 長軸方向の角度
    direction = eigvecs[:, 0]
    angle_rad = np.arctan2(direction[1], direction[0])

    # 楕円の軸方向は pi 周期なので [-pi/2, pi/2) に正規化
    angle_rad = (angle_rad + np.pi / 2) % np.pi - np.pi / 2

    return (float(center[0]), float(center[1]), float(a), float(b), float(angle_rad))


def plot_ellipse(cx, cy, a, b, angle_rad):
    """
    楕円 (cx, cy, a, b, angle_rad) を Matplotlib で描画する関数

    Parameters:
        cx, cy     : 楕円の中心座標
        a, b       : 長半径, 短半径 (a >= b)
        angle_rad  : x軸に対する回転角（ラジアン）
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # ラジアンから角度（度）に変換
    angle_deg = np.degrees(angle_rad)

    # matplotlib.patches.Ellipse を作成
    # ※ Ellipse の width, height は「直径 (2a, 2b)」を指定することに注意
    ellipse = Ellipse(
        xy=(cx, cy),
        width=2 * a,
        height=2 * b,
        angle=angle_deg,
        edgecolor="blue",
        facecolor="cyan",
        alpha=0.4,
        linewidth=2,
        label="Projected Ellipse",
    )

    # 描画オブジェクトに追加
    ax.add_patch(ellipse)

    # 中心点と主軸（長軸・短軸）のラインを描画
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 長軸ベクトル (長さ a)
    major_axis_x = [cx - a * cos_a, cx + a * cos_a]
    major_axis_y = [cy - a * sin_a, cy + a * sin_a]

    # 短軸ベクトル (長さ b)
    minor_axis_x = [cx - b * (-sin_a), cx + b * (-sin_a)]
    minor_axis_y = [cy - b * cos_a, cx + b * cos_a]  # 正確には cy + b * cos_a
    minor_axis_y = [cy - b * cos_a, cy + b * cos_a]

    ax.plot(major_axis_x, major_axis_y, "r--", linewidth=1.5, label="Major axis (a)")
    ax.plot(minor_axis_x, minor_axis_y, "g--", linewidth=1.5, label="Minor axis (b)")
    ax.plot(cx, cy, "ro", markersize=6, label="Center (cx, cy)")

    # グラフのスケールと表示範囲の設定
    # アスペクト比を 1:1 に固定して楕円が歪まないようにする
    ax.set_aspect("equal", adjustable="datalim")

    # 適切な描画範囲を設定
    max_r = max(a, b) * 1.5
    ax.set_xlim(cx - max_r, cx + max_r)
    ax.set_ylim(cy - max_r, cy + max_r)

    # ラベルとグリッド
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Ellipse: center=({cx:.2f}, {cy:.2f}), a={a:.2f}, b={b:.2f}, angle={angle_deg:.1f}°")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.show()


def get_orthonormal_vectors(ax: float, ay: float, az: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = np.array([ax, ay, az], dtype=float)
    norm = np.linalg.norm(a)
    if abs(a[0]) < 0.5 and abs(a[1]) < 0.5:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 0.0, 1.0])
    b = np.cross(a, helper)
    c = np.cross(a, b)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)
    return b / norm_b, c / norm_c


def project_3d_point(x: float, y: float, z: float):
    px = -x / z
    py = -y / z
    pz = -1.0
    return px, py, pz


#
# s: shrink_scale for plot, (x, y, z): center of the circle, (ax, ay, az) axis of the circle
#
def project_3d_circle_to_screen(img, s, x, y, z, ax, ay, az, r, n, rng=0.0, axis=False, flg_noise=False):
    nrnd = 13
    szrnd = 2.0
    h, w = img.shape
    w2 = w >> 1
    h2 = h >> 1
    v0, v1 = get_orthonormal_vectors(ax, ay, az)
    # print(v0, np.linalg.norm(v0))
    # print(v1, np.linalg.norm(v1))
    if axis:
        cv2.line(img, (0, h2), (w - 1, h2), 64, 1)
        cv2.line(img, (w2, 0), (w2, h - 1), 64, 1)
    vc = np.array([x, y, z])
    for k in range(n):
        t = (2 * k * np.pi) / n
        vp = r * (np.cos(t) * v0 + np.sin(t) * v1) + vc
        # print(vp)
        px, py, _ = project_3d_point(vp[0], vp[1], vp[2])
        # print(px, py)
        px *= s
        py *= s
        if 0.00001 < rng:
            px += np.random.uniform(-rng, rng)
            py += np.random.uniform(-rng, rng)
        ix = round(px) + w2
        iy = h2 - round(py)
        # print("(s * px, s * py)=(%8.3f, %8.3f)" % (s * px, s * py))
        img[iy, ix] = 255
        if flg_noise:
            irnd = np.random.randint(0, nrnd)
            if irnd == 0:
                dx0 = np.random.randint(-szrnd, szrnd)
                dx1 = np.random.randint(-szrnd, szrnd)
                dy0 = np.random.randint(-szrnd, szrnd)
                dy1 = np.random.randint(-szrnd, szrnd)
                cv2.line(img, (ix - dx0, iy - dy0), (ix + dx1, iy + dy1), 255)


#
# s: shrink_scale for plot, (x, y, z): center of the circle, (ax, ay, az) axis of the circle
#
def project_3d_circle_to_screen_rgb(img_rgb, s, x, y, z, ax, ay, az, r, n, color, rng=0.0, axis=False, flg_noise=False):
    nrnd = 13
    szrnd = 2.0
    h, w = img_rgb.shape[:2]
    w2 = w >> 1
    h2 = h >> 1
    v0, v1 = get_orthonormal_vectors(ax, ay, az)
    # print(v0, np.linalg.norm(v0))
    # print(v1, np.linalg.norm(v1))
    if axis:
        cv2.line(img_rgb, (0, h2), (w - 1, h2), (64, 64, 64), 1)
        cv2.line(img_rgb, (w2, 0), (w2, h - 1), (64, 64, 64), 1)
    vc = np.array([x, y, z])
    for k in range(n):
        t = (2 * k * np.pi) / n
        vp = r * (np.cos(t) * v0 + np.sin(t) * v1) + vc
        # print(vp)
        px, py, _ = project_3d_point(vp[0], vp[1], vp[2])
        # print(px, py)
        px *= s
        py *= s
        if 0.00001 < rng:
            px += np.random.uniform(-rng, rng)
            py += np.random.uniform(-rng, rng)
        ix = round(px) + w2
        iy = h2 - round(py)
        # print("(s * px, s * py)=(%8.3f, %8.3f)" % (s * px, s * py))
        cv2.line(img_rgb, (ix, iy), (ix + 1, iy), color, thickness=1)
        if flg_noise:
            irnd = np.random.randint(0, nrnd)
            if irnd == 0:
                dx0 = np.random.randint(-szrnd, szrnd)
                dx1 = np.random.randint(-szrnd, szrnd)
                dy0 = np.random.randint(-szrnd, szrnd)
                dy1 = np.random.randint(-szrnd, szrnd)
                cv2.line(img_rgb, (ix - dx0, iy - dy0), (ix + dx1, iy + dy1), 255)


def draw_ellipse_to_screen(img, s, cx, cy, a, b, angle_rad, n, rng=0.0, axis=False):
    lst_x = []
    lst_y = []
    h, w = img.shape
    w2 = w >> 1
    h2 = h >> 1
    cos_rad = np.cos(angle_rad)
    sin_rad = np.sin(angle_rad)
    if axis:
        cv2.line(img, (0, h2), (w - 1, h2), 64, 1)
        cv2.line(img, (w2, 0), (w2, h - 1), 64, 1)
    for k in range(2 * n):
        t = (k * np.pi) / n
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        _x = cx + a * cos_t * cos_rad - b * sin_t * sin_rad
        _y = cy + a * cos_t * sin_rad + b * sin_t * cos_rad
        lst_x.append(_x)
        lst_y.append(_y)
        x = s * _x
        y = s * _y
        if 0.0001 < rng:
            x += np.random.uniform(-rng, rng)
            y += np.random.uniform(-rng, rng)
        ix = round(x) + w2
        iy = h2 - round(y)
        # print("(%4d, %4d)" % (ix, iy))
        img[iy, ix] = 255
    return np.array(lst_x, dtype=np.float32), np.array(lst_y, dtype=np.float32)


def draw_ellipse_to_screen_rgb(img_rgb, s, cx, cy, a, b, angle_rad, n, color, rng=0.0, axis=False):
    lst_x = []
    lst_y = []
    h, w = img_rgb.shape[:2]
    w2 = w >> 1
    h2 = h >> 1
    cos_rad = np.cos(angle_rad)
    sin_rad = np.sin(angle_rad)
    if axis:
        cv2.line(img_rgb, (0, h2), (w - 1, h2), (64, 64, 64), 1)
        cv2.line(img_rgb, (w2, 0), (w2, h - 1), (64, 64, 64), 1)
    for k in range(2 * n):
        t = (k * np.pi) / n
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        _x = cx + a * cos_t * cos_rad - b * sin_t * sin_rad
        _y = cy + a * cos_t * sin_rad + b * sin_t * cos_rad
        lst_x.append(_x)
        lst_y.append(_y)
        x = s * _x
        y = s * _y
        if 0.0001 < rng:
            x += np.random.uniform(-rng, rng)
            y += np.random.uniform(-rng, rng)
        ix = round(x) + w2
        iy = h2 - round(y)
        # print("(%4d, %4d)" % (ix, iy))
        cv2.line(img_rgb, (ix, iy), (ix + 1, iy), color, thickness=1)


# ---------------------------------------------------------
# 動作確認用サンプルコード
# ---------------------------------------------------------
def simple_test():
    r = 0.5  # 半径
    z = -2.8
    ax, ay, az = -0.1, 1.0, 0.7  # 法線ベクトル（傾いた円）

    img = np.zeros((960, 1280), dtype=np.uint8)
    img2 = np.zeros((960, 1280), dtype=np.uint8)

    rpos = 1.2
    nframes = 100
    for k in range(nframes):
        t = (2 * k * np.pi) / nframes
        x = rpos * np.cos(t)
        y = rpos * np.sin(t)

        img.fill(0)
        project_3d_circle_to_screen(img, 500.0, x, y, z, ax, ay, az, r, 1000, 0.0, True)

        print(
            "(%8.3f, %8.3f, %8.3f) (%9.6f, %9.6f, %9.6f) r=%.3f: " % (x, y, z, ax, ay, az, r),
            end="",
        )
        cx, cy, a, b, angle_rad = get_parameters_of_ellipse(x, y, z, ax, ay, az, r)
        print("%8.3f, %8.3f, %8.3f, %8.3f, %9.6f" % (cx, cy, a, b, angle_rad))
        img2.fill(0)
        draw_ellipse_to_screen(img2, 500.0, cx, cy, a, b, angle_rad, 1000, 0.7, True)

        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
        # break

    cv2.destroyAllWindows()


def simple_test2():
    s = 900.0
    r = 0.5  # 半径
    x, y, z = -0.1, 0.7, -2.8
    ax, ay, az = -0.1, 1.0, 0.6  # 法線ベクトル（傾いた円）

    img = np.zeros((640, 480), dtype=np.uint8)
    img2 = np.zeros((640, 480), dtype=np.uint8)

    project_3d_circle_to_screen(img, s, x, y, z, ax, ay, az, r, 1000, 0.0, True)

    print(
        "(%8.3f, %8.3f, %8.3f) (%9.6f, %9.6f, %9.6f) r=%.3f: " % (x, y, z, ax, ay, az, r),
        end="",
    )
    cx, cy, a, b, angle_rad = get_parameters_of_ellipse(x, y, z, ax, ay, az, r)
    print("%8.3f, %8.3f, %8.3f, %8.3f, %9.6f" % (cx, cy, a, b, angle_rad))
    draw_ellipse_to_screen(img2, s, cx, cy, a, b, angle_rad, 300, 0.9, True)

    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_radius_a_and_b(r, d, t):
    c = np.cos(t)
    s = np.sin(t)
    q = d * d - r * r * s * s
    b = (r * d * c) / q
    a = (r * d) / np.sqrt(q)
    return a, b


def project_circle_to_ellipse(image_size, f, lens_pos, up_vec, circle_param):
    """
    XY平面上の円を透視投影し、画像平面上の楕円パラメータを計算します。

    Parameters
    ----------
    image_size : tuple of int
        (w, h) 画像の幅と高さ
    f : float
        焦電距離 (ピクセル単位)
    lens_pos : tuple of float
        (0, cy, cz) レンズ中心の位置 (3D世界座標)
    up_vec : tuple of float
        (ux, uy, uz) 画像の上方向ベクトル
    circle_param : tuple of float
        (u, v, r) XY平面上の円の中心(u, v)と半径r

    Returns
    -------
    dict or None
        楕円パラメータ:
        {
            'x': 中心x (ピクセル),
            'y': 中心y (ピクセル),
            'a': 半長軸 (長半径a),
            'b': 半短軸 (短半径b),
            'angle': 回転角 (度, x軸正の方向からの反時計回り)
        }
        投影結果が放物線・双曲線・画像範囲外等の場合は None
    """
    w, h = image_size
    cx_img, cy_img = w / 2.0, h / 2.0
    u_c, v_c, r = circle_param

    # 1. カメラ内部パラメータ行列 K
    K = np.array([[f, 0, cx_img], [0, f, cy_img], [0, 0, 1]], dtype=np.float64)

    # 2. カメラ外部パラメータ (Look-at 変換)
    C = np.array(lens_pos, dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # 視線方向 z_cam (カメラから原点を向く: Z軸の正方向を視線逆向きに定義)
    z_cam = C - target
    z_len = np.linalg.norm(z_cam)
    if z_len == 0:
        raise ValueError("カメラ位置が原点と一致しています。")
    z_cam /= z_len

    # 上方向の正規化と直交化
    up = np.array(up_vec, dtype=np.float64)
    x_cam = np.cross(up, z_cam)
    x_len = np.linalg.norm(x_cam)
    if x_len == 0:
        raise ValueError("上方向ベクトルが視線ベクトルと平行です。")
    x_cam /= x_len

    y_cam = np.cross(z_cam, x_cam)

    # 回転行列 R と 平行移動ベクトル t (世界座標 -> カメラ座標: P_cam = R * P_world + t)
    R = np.vstack([x_cam, y_cam, z_cam])  # 各行が基底
    t = -R @ C

    # 3. XY平面 (Z=0) から画像平面への 3x3 ホモグラフィ行列 H の導出
    # P_cam = R[:, 0]*X + R[:, 1]*Y + t
    # x_img ~ K * (R[:, 0]*X + R[:, 1]*Y + t)
    H = K @ np.hstack([R[:, 0:1], R[:, 1:2], t.reshape(3, 1)])

    # 4. 単位円 (x'^2 + y'^2 - 1 = 0) を (u_c, v_c, r) に移動する変換 M
    # [X, Y, 1]^T = M * [x', y', 1]^T
    M = np.array([[r, 0, u_c], [0, r, v_c], [0, 0, 1]], dtype=np.float64)

    # 単位円の二次形式行列 Q_unit = diag(1, 1, -1)
    Q_unit = np.diag([1.0, 1.0, -1.0])

    # 全体ホモグラフィ H_total = H * M
    H_total = H @ M
    H_inv = np.linalg.inv(H_total)

    # 画像平面上の二次形式行列 C_img = H_inv^T * Q_unit * H_inv
    C_img = H_inv.T @ Q_unit @ H_inv

    # 5. 二次形式 Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 から楕円パラメータの抽出
    A = C_img[0, 0]
    B = 2.0 * C_img[0, 1]
    C_val = C_img[1, 1]
    D = 2.0 * C_img[0, 2]
    E = 2.0 * C_img[1, 2]
    F = C_img[2, 2]

    # 判別式 (B^2 - 4AC < 0 で楕円)
    det = B**2 - 4.0 * A * C_val
    if det >= 0:
        return None  # 放物線または双曲線 (視野角を越えている等)

    # 楕円中心 (x0, y0)
    x0 = (2.0 * C_val * D - B * E) / det
    y0 = (2.0 * A * E - B * D) / det

    # 中心移動後の定数項 shrink_scale
    shrink_scale = A * x0**2 + B * x0 * y0 + C_val * y0**2 - F

    # 2x2 行列 M_2d の固有値・固有ベクトルによる軸長と回転角の算出
    M_2d = np.array([[A / shrink_scale, B / (2.0 * shrink_scale)], [B / (2.0 * shrink_scale), C_val / shrink_scale]])

    evals, evecs = np.linalg.eigh(M_2d)

    # 固有値は正である必要がある
    if np.any(evals <= 0):
        return None

    # 半軸長 (a >= b となるように整列)
    r1, r2 = 1.0 / np.sqrt(evals)
    if r1 >= r2:
        a, b = r1, r2
        main_evec = evecs[:, 0]
    else:
        a, b = r2, r1
        main_evec = evecs[:, 1]

    # 回転角 (ラジアン -> 度)
    angle_rad = np.arctan2(main_evec[1], main_evec[0])

    return [float(x0), float(y0), float(a), float(b), float(angle_rad)]


def point_to_ellipse_distance_single(x, y, a, b):
    """
    標準形の楕円 (x^2/a^2 + y^2/b^2 = 1) に対する
    点 (x, y) からの最短距離を計算 (ニュートン法/1次元最適化)
    """
    # 対称性を利用して第1象限に移動
    px, py = abs(x), abs(y)

    # 楕円上の点 (a*cos(t), b*sin(t)) との距離の二乗を最小化するパラメーター t を探索
    def distance_sq(t):
        ex = a * np.cos(t)
        ey = b * np.sin(t)
        return (px - ex) ** 2 + (py - ey) ** 2

    # 0 <= t <= pi/2 の範囲で最適化
    res = minimize_scalar(distance_sq, bounds=(0, np.pi / 2), method="bounded")
    return np.sqrt(res.fun)


def rotate_points(xs, ys, rad):
    """点群を原点周りに rad ラジアン回転させる（NumPy版）

    Parameters:
        xs (list or np.ndarray): X座標のリスト
        ys (list or np.ndarray): Y座標のリスト
        rad (float): 回転角（ラジアン）

    Returns:
        tuple: (回転後のX座標のリスト, 回転後のY座標のリスト)
    """

    cos_r = np.cos(rad)
    sin_r = np.sin(rad)

    x_rot = xs * cos_r - ys * sin_r
    y_rot = xs * sin_r + ys * cos_r

    # リストとして返したい場合は .tolist() を使用
    return x_rot, y_rot


def test_residuals():
    # 楕円パラメータ
    cx, cy, a, b, angle_rad = (
        -0.03514392227529725,
        0.2417169608039435,
        0.18071666605191075,
        0.05412232595395583,
        0.10271627768349534,
    )
    s = 900.0

    w = 480
    h = 640
    img = np.zeros((h, w), dtype=np.uint8)

    list_x32f, list_y32f = draw_ellipse_to_screen(img, s, cx, cy, a, b, angle_rad, 100)
    # print(list_x32f)
    # print(list_y32f)

    list_x32f -= cx
    list_y32f -= cy

    # print(list_x32f)
    # print(list_y32f)

    list_x32f_rot, list_y32f_rot = rotate_points(list_x32f, list_y32f, -angle_rad)
    # print(list_x32f_rot)
    # print(list_y32f_rot)

    # for k, x in enumerate(list_x32f_rot):
    #    y = list_y32f_rot[k]
    #    print("%8.2f, %8.2f" % (1000 * x, 1000 * y))
    # exit()

    residuals = np.sum((list_x32f_rot / a) ** 2 + (list_y32f_rot / b) ** 2 - 1.0)
    sse = np.sum(residuals**2)
    print(sse)


def test_residuals2():
    # 楕円パラメータ
    cx, cy, a, b, angle_rad = (
        -0.03514392227529725,
        0.2417169608039435,
        0.18071666605191075,
        0.05412232595395583,
        0.10271627768349534,
    )
    s = 900.0

    w = 480
    h = 640
    # w = 1920
    # h = 1080
    img = np.zeros((h, w), dtype=np.uint8)

    draw_ellipse_to_screen(img, s, cx, cy, a, b, angle_rad, 100)

    cv2.imshow("Image 2", img)
    cv2.waitKey(0)

    h, w = img.shape
    w2 = w >> 1
    h2 = h >> 1
    list_y, list_x = np.where(img == 255)
    # print(list_x)
    # print(list_y)
    list_x -= w2
    list_y = h2 - list_y

    list_x32f = list_x.astype(np.float32)
    list_y32f = list_y.astype(np.float32)

    list_x32f /= s
    list_y32f /= s

    list_x32f -= cx
    list_y32f -= cy

    # print(list_x32f)
    # print(list_y32f)

    list_x32f_rot, list_y32f_rot = rotate_points(list_x32f, list_y32f, -angle_rad)
    # print(list_x32f_rot)
    # print(list_y32f_rot)

    # for k, x in enumerate(list_x32f_rot):
    #    y = list_y32f_rot[k]
    #    print("%8.2f, %8.2f" % (1000 * x, 1000 * y))
    # exit()

    npx = np.array(list_x32f_rot)
    npy = np.array(list_y32f_rot)

    residuals = np.sum((npx / a) ** 2 + (npy / b) ** 2 - 1.0)
    sse = np.sum(residuals**2)
    print(sse)


def test_residuals3():
    s = 900.0
    r = 0.5  # 半径
    x, y, z = -0.1, 0.7, -2.8
    ax, ay, az = -0.1, 1.0, 0.6  # 法線ベクトル（傾いた円）

    cx, cy, a, b, angle_rad = get_parameters_of_ellipse(x, y, z, ax, ay, az, r)
    print("cx, cy, a, b, angle_rad =", cx, cy, a, b, angle_rad)

    img = np.zeros((640, 480), dtype=np.uint8)

    project_3d_circle_to_screen(img, s, x, y, z, ax, ay, az, r, 100, 0.8, True)

    cv2.imshow("Image 3", img)
    cv2.waitKey(0)

    h, w = img.shape
    w2 = w >> 1
    h2 = h >> 1
    list_y, list_x = np.where(img == 255)
    # print(list_x)
    # print(list_y)
    list_x -= w2
    list_y = h2 - list_y

    list_x32f = list_x.astype(np.float32)
    list_y32f = list_y.astype(np.float32)

    list_x32f /= s
    list_y32f /= s

    list_x32f -= cx
    list_y32f -= cy

    # print(list_x32f)
    # print(list_y32f)

    list_x32f_rot, list_y32f_rot = rotate_points(list_x32f, list_y32f, -angle_rad)

    # print(list_x32f_rot)
    # print(list_y32f_rot)
    # for k, x in enumerate(list_x32f_rot):
    #    y = list_y32f_rot[k]
    #    print("%8.2f, %8.2f" % (1000 * x, 1000 * y))

    npx = np.array(list_x32f_rot)
    npy = np.array(list_y32f_rot)

    residuals = np.sum((npx / a) ** 2 + (npy / b) ** 2 - 1.0)
    sse = np.sum(residuals**2)
    print(sse)


def eee(x, y, z, ax, az, s):
    print("eee(x, y, z, ax, az, s)=(%9.6f, %9.6f, %9.6f, %9.6f, %9.6f, %9.2f)" % (x, y, z, ax, az, s))
    """
    3D 円を z=-1 平面に透視投影した楕円のパラメーターを返す。

    円の中心: (x, y, z)
    円の軸方向: (ax, 1.0, az)
    円の半径: 1.0
    カメラ中心: 原点
    投影式: u = -X/Z, v = -Y/Z

    Returns:
        ex, ey       : 楕円の中心
        ea, eb       : 半長軸、半短軸 (ea >= eb)
        angle_rad    : 長軸の角度（ラジアン）
    """

    # 円の軸方向を正規化
    n = np.array([ax, 1.0, az], dtype=np.float64)
    n /= np.linalg.norm(n)

    center = np.array([x, y, z], dtype=np.float64)

    # 円の平面がカメラ中心を通る場合は退化
    if abs(np.dot(n, center)) < 1e-12:
        raise ValueError("円の平面がカメラ中心を通るため、投影が退化します")

    # 円の平面内の正規直交基底を作る
    ref = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(n, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    # 円周のパラメーター s² + t² = 1
    # X = x + e1[0]*s + e2[0]*t
    # Y = y + e1[1]*s + e2[1]*t
    # Z = z + e1[2]*s + e2[2]*t
    #
    # 透視投影: u = -X/Z, v = -Y/Z
    H = np.array([[e1[0], e2[0], x], [e1[1], e2[1], y], [-e1[2], -e2[2], -z]], dtype=np.float64)

    # 単位円 s² + t² - 1 = 0
    C = np.diag([1.0, 1.0, -1.0])

    # 投影後の円錐曲線行列
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise ValueError("投影変換が退化しています")

    Q = Hinv.T @ C @ Hinv
    Q = (Q + Q.T) / 2.0

    # 二次形式: p.T @ A @ p + 2*d.T @ p + f = 0
    A = Q[:2, :2]
    d = Q[:2, 2]
    f = Q[2, 2]

    try:
        ex, ey = -np.linalg.solve(A, d)
    except np.linalg.LinAlgError:
        raise ValueError("楕円の中心を計算できません")

    c = np.array([ex, ey])
    Fc = c @ A @ c + 2.0 * d @ c + f

    # 固有値・固有ベクトルから半軸長を計算
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    lengths2 = -Fc / eigenvalues

    if np.any(lengths2 <= 0):
        raise ValueError("投影結果が実楕円になりません")

    lengths = np.sqrt(lengths2)

    # 半長軸 ea >= 半短軸 eb
    order = np.argsort(lengths)[::-1]
    ea = lengths[order[0]]
    eb = lengths[order[1]]

    # 長軸の角度
    v = eigenvectors[:, order[0]]
    angle_rad = np.arctan2(v[1], v[0])

    # [-pi/2, pi/2) に正規化
    angle_rad = (angle_rad + np.pi / 2) % np.pi - np.pi / 2

    print(">>> %8.4f, %8.4f, %8.4f, %8.4f, %9.6f" % (ex, ey, ea, eb, angle_rad))

    return float(s * ex), float(s * ey), float(s * ea), float(s * eb), float(angle_rad)


def transform_points(xs, ys, cx, cy, angle_rad):
    # 中心移動
    dx = xs - cx
    dy = ys - cy
    # 回転変換 (時計回り)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a
    return x_rot, y_rot


g_idx = 0

g_w, g_h = 480, 640
g_scale = 900.0
g_rcoil = 1.0
g_x, g_y, g_z = -0.2, 1.2, -5.6
g_ax, g_ay, g_az = 0.1, 1.0, 0.6  # 法線ベクトル（傾いた円）
g_ratio = 0.714


def debug_plot2(xs, ys, s, w, h):
    w2 = w >> 1
    h2 = h >> 1
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.line(img, (0, h2), (w - 1, h2), 63, 1)
    cv2.line(img, (w2, 0), (w2, h - 1), 63, 1)
    cv2.line(img, (int(s) + w2, h2 - 5), (int(s) + w2, h2 + 5), 63, 1)
    cv2.line(img, (-int(s) + w2, h2 - 5), (-int(s) + w2, h2 + 5), 63, 1)
    cv2.line(img, (w2 - 5, h2 + int(s)), (w2 + 5, h2 + int(s)), 63, 1)
    cv2.line(img, (w2 - 5, h2 - int(s)), (w2 + 5, h2 - int(s)), 63, 1)
    for k, x in enumerate(xs):
        y = ys[k]
        # print("(%8.6f, %8.6f)" % (x, y))
        ix = round(s * x) + w2
        iy = h2 - round(s * y)
        # print("(%4d, %4d)" % (ix, iy))
        img[iy, ix] = 255
    return img


def test_ellipse_fit():
    global g_idx

    """
    img = np.zeros((g_h, g_w), dtype=np.uint8)
    project_3d_circle_to_screen(img, g_scale, g_x, g_y, g_z, g_ax, g_ay, g_az, g_rcoil, 1000, 0.0, True)
    cv2.imshow("img", img)

    binary_img = np.zeros((g_h, g_w), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img, g_scale, g_x, g_y, g_z, g_ax, g_ay, g_az, g_rcoil, 1000, 1.7, False, True)
    binary_img2 = np.zeros((g_h, g_w), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img2, g_scale, g_x, g_y, g_z, g_ax, g_ay, g_az, g_ratio * g_rcoil, 500, 1.7, False, True)

    cv2.imwrite(g_path_img_ellipse_0, binary_img)
    cv2.imwrite(g_path_img_ellipse_1, binary_img2)
    """

    binary_img = cv2.imread(g_path_img_ellipse_0, cv2.IMREAD_GRAYSCALE)
    binary_img2 = cv2.imread(g_path_img_ellipse_1, cv2.IMREAD_GRAYSCALE)
    binary_img_or = binary_img | binary_img2
    cv2.imwrite(g_path_img_ellipse, binary_img_or)

    # cv2.imshow("binary_img", binary_img)
    # cv2.imshow("binary_img2", binary_img2)
    # cv2.waitKey(0)

    h, w = binary_img.shape
    w2 = w >> 1
    h2 = h >> 1

    ys, xs = np.where(0 < binary_img)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    xs -= w2
    ys = h2 - ys

    ys2, xs2 = np.where(0 < binary_img2)
    xs2 = np.array(xs2, dtype=np.float64)
    ys2 = np.array(ys2, dtype=np.float64)
    xs2 -= w2
    ys2 = h2 - ys2

    img_gt = debug_plot2(xs, ys, 1.0, g_w, g_h)
    img_gt2 = debug_plot2(xs2, ys2, 1.0, g_w, g_h)
    img_gt_or = img_gt | img_gt2
    # cv2.imshow("img_gt", img_gt)
    # cv2.imshow("img_gt2", img_gt2)
    # cv2.imshow("img_gt_or", img_gt_or)
    # cv2.waitKey(0)

    shrink_scale = 1200.0
    # shrink_scale = 800.0
    # shrink_scale = g_scale

    ds = shrink_scale / g_scale

    xs /= shrink_scale
    ys /= shrink_scale
    xs2 /= shrink_scale
    ys2 /= shrink_scale

    xy = []
    for k, xx in enumerate(xs):
        yy = ys[k]
        xy.append((xx, yy))

    xy2 = []
    for k, xx in enumerate(xs2):
        yy = ys2[k]
        xy2.append((xx, yy))

    # x  = -0.19224232418766282
    # y  = 1.4296498320918074
    # z  = -3.9609736429605626

    # initial = (g_x, g_y, g_z, g_ax, g_az)
    initial = (ds * g_x, ds * g_y, ds * g_z, 0.0, 0.5)

    x, y, z, ax, az, result = fit_circle_from_projected_points(xy, xy2, initial, n_starts=8)

    print("成功:", result.success)
    print("終了理由:", result.message)
    print("終了ステータス:", result.status)
    print("関数評価回数:", result.nfev)
    print("最終コスト:", result.cost)
    print("最適性:", result.optimality)

    """
    x = g_x
    y = g_y
    z = g_z
    ax = g_ax
    az = g_az
    """

    print("x  =", x)
    print("y  =", y)
    print("z  =", z)
    print("ax =", ax)
    print("az =", az)

    cv2.waitKey(0)

    img_gt_or_rgb = cv2.cvtColor(img_gt_or, cv2.COLOR_GRAY2RGB)
    project_3d_circle_to_screen_rgb(img_gt_or_rgb, shrink_scale, x, y, z, ax, 1.0, az, 1.0, 1200, (0, 255, 0), 0.0, True)
    project_3d_circle_to_screen_rgb(img_gt_or_rgb, shrink_scale, x, y, z, ax, 1.0, az, g_ratio, 600, (0, 0, 255))
    cv2.imshow("img_gt_or_rgb", img_gt_or_rgb)

    cv2.waitKey(0)


def conic_to_ellipse(Q):
    """
    3x3 の楕円行列 Q から楕円パラメータを計算する。

    楕円の方程式:
        [u, v, 1] @ Q @ [u, v, 1].T = 0

    Returns:
        ex, ey      : 楕円の中心
        ea, eb      : 長半径、短半径 (ea >= eb)
        angle_rad   : 長軸と X 軸のなす角度 [rad]
                      範囲は [-pi/2, pi/2)

    Raises:
        ValueError: Q が有効な楕円を表していない場合
    """
    Q = np.asarray(Q, dtype=np.float64)

    if Q.shape != (3, 3):
        raise ValueError("Q は 3x3 行列である必要があります")

    if not np.all(np.isfinite(Q)):
        raise ValueError("Q に NaN または inf が含まれています")

    # 数値誤差による非対称性を除去
    Q = (Q + Q.T) * 0.5

    # 楕円の方程式:
    # A の部分が二次項、b が一次項、c が定数項
    A = Q[:2, :2]
    b = Q[:2, 2]
    c = Q[2, 2]

    # 中心を求める:
    # A @ center + b = 0
    try:
        center = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        raise ValueError("楕円の中心を計算できません")

    ex, ey = center

    # 中心を原点に移動したときの定数項
    f0 = center @ A @ center + 2.0 * b @ center + c

    # 固有値の符号が負の場合、Q 全体の符号を反転する
    eigvals = np.linalg.eigvalsh(A)

    if np.all(eigvals < 0.0):
        A = -A
        f0 = -f0
    elif not np.all(eigvals > 0.0):
        raise ValueError("Q は実楕円を表していません")

    # 中心移動後の方程式が楕円になることを確認
    if f0 >= 0.0:
        raise ValueError("Q は実楕円を表していません")

    # A の固有値・固有ベクトル
    eigvals, eigvecs = np.linalg.eigh(A)

    # 半径:
    # lambda1 * w1^2 + lambda2 * w2^2 = -f0
    radii = np.sqrt(-f0 / eigvals)

    # eigh は固有値を昇順に返すため、
    # 半径は降順（長半径、短半径）になる
    ea = float(radii[0])
    eb = float(radii[1])

    # 最大半径に対応する固有ベクトルが長軸方向
    vx, vy = eigvecs[:, 0]
    angle_rad = float(np.arctan2(vy, vx))

    # 角度を [-pi/2, pi/2) に正規化
    angle_rad = (angle_rad + np.pi / 2) % np.pi - np.pi / 2

    print("conic_to_ellipse: (ex, ey, ea, eb, rad)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (float(ex), float(ey), ea, eb, angle_rad))

    return float(ex), float(ey), ea, eb, angle_rad


def fit_circle_from_projected_points(xy, xy2, initial=None, n_starts=1):
    """
    z=-1 平面上の点群に投影楕円が最もよく合うように、
    元の半径1の円のパラメーターを推定する。

    Parameters
    ----------
    xy : array-like, shape (N, 2)
        投影平面上の点群 [(x0,y0), (x1,y1), ...]
    initial : tuple or None
        初期値 (x, y, z, ax, az)
    n_starts : int
        初期値を変えて探索する回数

    Returns
    -------
    x, y, z, ax, az : float
        推定された円の中心と軸方向パラメーター

    Notes
    -----
    円の中心は (x,y,z)、半径は1。
    軸方向は (ax,1,az)。
    カメラ中心は原点、投影式は u=-X/Z, v=-Y/Z。
    """

    xy = np.asarray(xy, dtype=np.float64)
    xy2 = np.asarray(xy2, dtype=np.float64)

    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 5:
        raise ValueError("xy は5点以上の (N, 2) 配列にしてください")

    if xy2.ndim != 2 or xy2.shape[1] != 2 or len(xy2) < 5:
        raise ValueError("xy2 は5点以上の (N, 2) 配列にしてください")

    if not np.all(np.isfinite(xy)):
        raise ValueError("xy に NaN または inf が含まれています")

    if not np.all(np.isfinite(xy2)):
        raise ValueError("xy2 に NaN または inf が含まれています")

    # 点群の概略中心
    mean_xy = np.mean(xy, axis=0)

    # 円の平面内の正規直交基底を作る
    def make_conic2(params, r):
        x, y, z, ax, az = params

        n = np.array([ax, 1.0, az], dtype=np.float64)
        n /= np.linalg.norm(n)

        # center = np.array([x, y, z], dtype=np.float64)

        # 円の平面内の正規直交基底
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(n, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])

        e1 = np.cross(n, ref)
        e1 /= np.linalg.norm(e1)

        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)

        # 円周の同次座標から投影座標への変換
        H = np.array([[e1[0], e2[0], x], [e1[1], e2[1], y], [-e1[2], -e2[2], -z]])

        if abs(np.linalg.det(H)) < 1e-12:
            raise np.linalg.LinAlgError("投影が退化しています")

        Hinv = np.linalg.inv(H)

        # s² + t² - 1 = 0
        C = np.diag([1.0, 1.0, -1.0])
        Q = Hinv.T @ C @ Hinv
        Q_0 = (Q + Q.T) * 0.5

        # 半径 r の円の方程式: s² + t² - r² = 0
        C2 = np.diag([1.0, 1.0, -(r**2)])
        Q2 = Hinv.T @ C2 @ Hinv
        Q_1 = (Q2 + Q2.T) * 0.5

        return Q_0, Q_1, n

    # 点群から円錐曲線までの近似距離（Sampson distance）
    def residual(params):
        global g_idx
        # print("g_idx:", g_idx)
        # print("params:", params)
        # exit()

        try:
            Q, Q2, n = make_conic2(params, g_ratio)
            # conic_to_ellipse(Q)
            # conic_to_ellipse(Q2)
            # exit()
        except (np.linalg.LinAlgError, FloatingPointError):
            raise ValueError("Q, Q2, n = make_conic2(params, g_ratio)")
            return np.full(len(xy) + 1, 1e3)

        points = np.column_stack([xy, np.ones(len(xy))])
        Qp = points @ Q.T
        F = np.sum(points * Qp, axis=1)
        # 楕円の法線方向の距離に近い誤差
        grad = 2.0 * Qp[:, :2]
        grad_norm = np.linalg.norm(grad, axis=1)
        dist = F / np.maximum(grad_norm, 1e-12)

        points2 = np.column_stack([xy2, np.ones(len(xy2))])
        Qp2 = points2 @ Q2.T
        F2 = np.sum(points2 * Qp2, axis=1)
        # 楕円の法線方向の距離に近い誤差
        grad2 = 2.0 * Qp2[:, :2]
        grad_norm2 = np.linalg.norm(grad2, axis=1)
        dist2 = F2 / np.maximum(grad_norm2, 1e-12)

        # 円周全体がカメラ前方 Z<0 に存在する条件
        # 円のZ方向の半径は sqrt(1 - n_z²)
        z_extent = np.sqrt(max(0.0, 1.0 - n[2] ** 2))
        visibility_error = max(0.0, params[2] + z_extent + 1e-6) * 10.0

        g_idx += 1

        # print("(Q, n):", Q, n)
        # print("Qp:", Qp)
        # print("F:", F)
        # print("F2:", F2)
        # print(F.min(), F.max())

        # print("dist:", dist)
        # print("dist2:", dist2)
        # print("visibility_error:", visibility_error)

        # avg_dist = np.mean(dist)
        # return np.append(avg_dist, visibility_error)

        # exit()
        # return np.append(dist, visibility_error)
        # return np.append(dist2, visibility_error)
        return np.append(dist, np.append(dist2, visibility_error))
        # return np.append(dist, dist2, visibility_error)

        # dist_mod = dist[::2]
        # return np.append(dist_mod, visibility_error)

    if initial is not None:
        starts = np.asarray(initial, dtype=float)
    else:
        raise RuntimeError("initial is None")

    lower = [-1.0, 0.0, -20.0, -0.5, 0.0]
    upper = [1.0, 5.0, -1.0, 0.5, 1.0]

    best_result = None
    best_cost = np.inf

    for start in starts:
        # print("    start:", start)
        p0 = start.copy()
        # print("    p0:", p0)
        # print("    p0[2]:", p0[2])
        # p0[2] = min(p0[2], -1.0)
        # p0[2] = min(p0[2], -1e-4)
        # exit()

        # check bounds
        p0_arr = np.array(p0)
        is_valid = (p0_arr >= lower) & (p0_arr <= upper)
        if not np.all(is_valid):
            print("範囲外の初期値があります！")
            print("p0:", p0_arr)
            print("下限:", lower)
            print("上限:", upper)
            print("有効な要素 (True/False):", is_valid)
            exit()

        result = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.05,  # 0.05, 0.01
            max_nfev=25,  # ???
            verbose=0,
        )

        """
        print("成功:", result.success)
        print("終了理由:", result.message)
        print("終了ステータス:", result.status)
        print("関数評価回数:", result.nfev)
        print("最終コスト:", result.cost)
        print("最適性:", result.optimality)
        """

        err = np.mean(residual(result.x)[:-1] ** 2)

        if err < best_cost:
            best_cost = err
            best_result = result

    if best_result is None:
        raise RuntimeError("最適化に失敗しました")

    x, y, z, ax, az = best_result.x

    # 収束状況を確認したい場合は result.success や best_cost を調べる
    return float(x), float(y), float(z), float(ax), float(az), result


def test_residuals4():
    s = 900.0
    r = 0.5  # 半径
    x, y, z = -0.1, 0.7, -2.8
    ax, ay, az = -0.1, 1.0, 0.6  # 法線ベクトル（傾いた円）

    s = 10000
    x, y, z = 0.0, 0.0, -10.0
    ax, ay, az = 0.0, 1.0, 0.5  # 法線ベクトル（傾いた円）

    img = np.zeros((1080, 1920), dtype=np.uint8)

    project_3d_circle_to_screen(img, s, x, y, z, ax, ay, az, r, 100)

    cv2.imshow("Image 3", img)
    cv2.waitKey(0)

    params = fit_ellipse_least_squares(img)
    if params is None:
        return

    x = params[0]
    y = params[1]
    a = params[2]
    b = params[3]
    angle_rad = params[4]
    print("(x, y, a, b, angle_rad):", x, y, a, b, angle_rad)

    h, w = img.shape
    w2 = w >> 1
    h2 = h >> 1
    x -= w2
    y = h2 - y
    angle_rad = -angle_rad
    print("(x, y, a, b, angle_rad):", x, y, a, b, angle_rad)

    rad = np.arccos(b / a)
    print("rad", rad)

    npc, nps = np.cos(angle_rad), np.sin(angle_rad)
    v = np.array([0.0, 0.0, 1.0])
    va = np.array([npc, -nps, 0.0])
    vrot = rotate_vector(v, va, rad)
    print("v:", v)
    print("va:", va)
    print("vrot:", vrot)


def test_ellipse_fit2():
    global g_idx

    img = np.zeros((g_h, g_w), dtype=np.uint8)

    binary_img = np.zeros((g_h, g_w), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img, g_scale, g_x, g_y, g_z, g_ax, g_ay, g_az, g_rcoil, 1000, 1.7, False, True)
    cv2.imshow("binary_img", binary_img)

    cv2.imwrite(g_path_img_ellipse_0, binary_img)
    """
    """

    binary_img = cv2.imread(g_path_img_ellipse_0, cv2.IMREAD_GRAYSCALE)

    ys, xs = np.where(0 < binary_img)

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    xs -= g_w >> 1
    ys = (g_h >> 1) - ys
    xs /= g_scale
    ys /= g_scale
    # print(xs)
    # print(ys)
    # exit()

    ex, ey, ea, eb, angle_rad = fit_ellipse(xs, ys)

    print("fit_ellipse_from_gray: (ex, ey, ea, eb, angle_rad)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (ex, ey, ea, eb, angle_rad))

    draw_ellipse_to_screen(img, g_scale, ex, ey, ea, eb, angle_rad, 2000, 0.0, True)
    cv2.imshow("img", img)

    x, y, z, ax, az = ellipse_2d_to_3d(ex, ey, ea, eb, angle_rad)

    print("ellipse_2d_to_3d: (x, y, z, ax, az)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (x, y, z, ax, az))

    cv2.waitKey(0)


def fit_ellipse(xs, ys):
    points = np.column_stack((xs, ys)).astype(np.float32)
    # print("points:", points)
    ellipse = cv2.fitEllipse(points)
    # print("ellipse:", ellipse)
    (ex, ey), (w, h), degree = ellipse
    return float(ex), float(ey), w / 2, h / 2, float(np.radians(degree))


def _fit_ellipse(xs, ys):
    """
    点群 (xs, ys) から楕円をフィッティングする。

    Parameters
    ----------
    xs, ys : list または numpy.ndarray
        楕円上の点の X 座標、Y 座標。

    Returns
    -------
    ex, ey, ea, eb, erad : tuple of float
        ex, ey : 楕円の中心座標
        ea     : 長半径
        eb     : 短半径 (ea >= eb)
        erad   : 長軸と X 軸のなす角度 [rad]
                 範囲 [-pi/2, pi/2)

    Notes
    -----
    楕円の全周に近い点群を想定する。
    外れ値の影響を軽減するため soft_l1 損失を使用する。
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys):
        raise ValueError("xs と ys は同じ長さの一次元配列にしてください")

    points = np.column_stack((xs, ys))
    if len(points) < 5:
        raise ValueError("楕円のフィッティングには少なくとも5点が必要です")
    if not np.all(np.isfinite(points)):
        raise ValueError("座標に NaN または inf が含まれています")

    # 点群の平均と共分散行列から初期値を推定
    center0 = np.mean(points, axis=0)
    centered = points - center0
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] <= 1e-12 or not np.all(np.isfinite(eigvals)):
        raise ValueError("点群が退化しているため楕円を推定できません")

    # 固有値の大きい方向を長軸とする
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 楕円周上に均等に分布する点なら、
    # 共分散の固有値は半径の二乗の約1/2
    a0, b0 = np.sqrt(2.0 * eigvals)
    theta0 = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # パラメータ: 中心X, 中心Y, log(長半径), log(短半径), 回転角
    p0 = np.array([center0[0], center0[1], np.log(a0), np.log(b0), theta0])

    def residual(p):
        ex, ey, loga, logb, theta = p
        a = np.exp(loga)
        b = np.exp(logb)

        dx = xs - ex
        dy = ys - ey

        ct = np.cos(theta)
        st = np.sin(theta)

        # 点を楕円のローカル座標系に変換
        X = ct * dx + st * dy
        Y = -st * dx + ct * dy

        # 楕円の方程式 F = 0
        F = (X / a) ** 2 + (Y / b) ** 2 - 1.0

        # 勾配の大きさで正規化し、距離に近い残差にする
        gx = 2.0 * X / (a * a)
        gy = 2.0 * Y / (b * b)

        grad_norm = np.sqrt(gx * gx + gy * gy)
        return F / np.maximum(grad_norm, 1e-12)

    result = least_squares(residual, p0, loss="soft_l1", f_scale=max(np.sqrt(np.mean(eigvals)), 1e-6), max_nfev=30)

    if not result.success and not np.all(np.isfinite(result.x)):
        raise ValueError("楕円のフィッティングに失敗しました")

    ex, ey, loga, logb, erad = result.x
    ea = float(np.exp(loga))
    eb = float(np.exp(logb))

    # 長半径と短半径の順序を保証する
    if ea < eb:
        ea, eb = eb, ea
        erad += np.pi / 2.0

    # 角度を [-pi/2, pi/2) に正規化
    erad = (erad + np.pi / 2.0) % np.pi - np.pi / 2.0

    return float(ex), float(ey), ea, eb, float(erad)


def ellipse_2d_to_3d(ex, ey, ea, eb, angle_rad):
    """
    2D楕円から半径1の3D円を推定する。

    Returns
    -------
    x, y, z, ax, az
        3D円の中心: (x, y, z)
        3D円の軸方向: (ax, 1.0, az)

    制約:
        z < 0
        az > 0
    """
    values = np.array([ex, ey, ea, eb, angle_rad], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("入力に NaN または inf が含まれています")
    if ea <= 0 or eb <= 0:
        raise ValueError("ea と eb は正である必要があります")

    # 長半径 ea、短半径 eb に統一
    if ea < eb:
        ea, eb = eb, ea
        angle_rad += np.pi / 2

    angle_rad = (angle_rad + np.pi / 2) % np.pi - np.pi / 2

    # 入力された楕円上に点を生成
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    ct = np.cos(angle_rad)
    st = np.sin(angle_rad)

    u = ex + ea * np.cos(t) * ct - eb * np.sin(t) * st
    v = ey + ea * np.cos(t) * st + eb * np.sin(t) * ct

    points = np.column_stack([u, v, np.ones_like(u)])

    def make_conic(p):
        x, y, z, ax, az = p

        n = np.array([ax, 1.0, az], dtype=float)
        n /= np.linalg.norm(n)

        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(n, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])

        e1 = np.cross(n, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)

        H = np.array([[e1[0], e2[0], x], [e1[1], e2[1], y], [-e1[2], -e2[2], -z]])

        if abs(np.linalg.det(H)) < 1e-12:
            raise np.linalg.LinAlgError("退化した投影です")

        Hinv = np.linalg.inv(H)
        C = np.diag([1.0, 1.0, -1.0])

        Q = Hinv.T @ C @ Hinv
        return (Q + Q.T) * 0.5, n

    def residual(p):
        try:
            Q, n = make_conic(p)

            Qp = points @ Q.T
            F = np.sum(points * Qp, axis=1)

            grad = 2.0 * Qp[:, :2]
            grad_norm = np.linalg.norm(grad, axis=1)

            # 楕円までの近似符号付き距離
            dist = F / np.maximum(grad_norm, 1e-12)

            # 円全体が Z < 0 に存在するための制約
            z_extent = np.sqrt(max(0.0, 1.0 - n[2] ** 2))
            visibility_error = max(0.0, p[2] + z_extent + 1e-6) * 10.0

            if not np.all(np.isfinite(dist)):
                raise FloatingPointError()

            return np.append(dist, visibility_error)

        except (np.linalg.LinAlgError, FloatingPointError):
            return np.full(len(points) + 1, 1e3)

    # z < 0、az > 0 を境界条件で保証
    lower = [-1.0, 0.0, -20.0, -0.5, 0.0]
    upper = [1.0, 5.0, -1.0, 0.5, 1.0]

    scale = max(ea, eb)

    # 複数の初期値から探索し、最も残差が小さい解を採用
    best = None

    depth = 1.0 / scale
    ax0 = 0.0
    az0 = 0.5
    p0 = np.array([ex * depth, ey * depth, -depth, ax0, az0])

    # check bounds
    x0_arr = np.array(p0)
    is_valid = (x0_arr >= lower) & (x0_arr <= upper)
    if not np.all(is_valid):
        print("範囲外の初期値があります！")
        print("x0:", x0_arr)
        print("下限:", lower)
        print("上限:", upper)
        print("有効な要素 (True/False):", is_valid)
        exit()

    result = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(min(ea, eb) * 0.02, 1e-6),
        max_nfev=30,
    )

    r = residual(result.x)
    cost = np.sum(r**2)

    if np.all(np.isfinite(result.x)):
        if best is None or cost < best[0]:
            best = (cost, result.x)

    if best is None:
        raise ValueError("3D円の推定に失敗しました")

    cost, p = best
    rms = np.sqrt(np.mean(residual(p)[:-1] ** 2))

    x, y, z, ax, az = p

    if rms > 0.01 * min(ea, eb):
        x, y, z, ax, az = 0.0, 0.0, -10.0, 0.0, 1.0
        # raise ValueError(f"入力楕円に合う解を十分な精度で求められませんでした。 近似誤差 RMS={rms:.6g}")

    assert z < 0
    assert az > 0

    return float(x), float(y), float(z), float(ax), float(az)


def _test_ellipse_fit3(x0, y0, z0, ax0, az0):
    wimg = 800
    himg = 800
    binary_img = np.zeros((himg, wimg), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img, g_scale, x0, y0, z0, ax0, 1.0, az0, g_rcoil, 1000, 1.7, False, True)
    binary_img2 = binary_img.copy()

    ys, xs = np.where(0 < binary_img)

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    w2 = wimg >> 1
    h2 = himg >> 1
    xs -= w2
    ys = h2 - ys
    xs /= g_scale
    ys /= g_scale

    ex, ey, ea, eb, angle_rad = fit_ellipse(xs, ys)

    print("fit_ellipse_from_gray: (ex, ey, ea, eb, angle_rad)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (ex, ey, ea, eb, angle_rad))

    img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    draw_ellipse_to_screen_rgb(img_rgb, g_scale, ex, ey, ea, eb, angle_rad, 2000, (0, 255, 0), 0.0, True)
    cv2.imshow("img_rgb", img_rgb)

    x, y, z, ax, az = ellipse_2d_to_3d(ex, ey, ea, eb, angle_rad)
    print("ellipse_2d_to_3d: (x, y, z, ax, az)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (x, y, z, ax, az))

    img_rgb2 = cv2.cvtColor(binary_img2, cv2.COLOR_GRAY2RGB)
    project_3d_circle_to_screen_rgb(img_rgb2, g_scale, x, y, z, ax, 1.0, az, g_rcoil, 1000, (0, 0, 255), 0.0, True)
    cv2.imshow("img_rgb2", img_rgb2)

    cv2.waitKey(10)


def calc_ellipse_line_intersection(ex, ey, ea, eb, erad):
    """
    中心 (ex, ey)、半径 (ea, eb)、回転角 erad の楕円と、
    (ex, ey) を通り方向 (-sin(erad), cos(erad)) の直線との交点を求める関数。

    Parameters:
        ex, ey (float): 楕円の中心座標
        ea (float): 半長径
        eb (float): 半短径
        erad (float): 回転角 (ラジアン)

    Returns:
        tuple: 交点1と交点2の座標 ((x1, y1), (x2, y2))
    """
    sin_val = np.sin(erad)
    cos_val = np.cos(erad)

    # 交点の座標を計算
    x1 = ex - eb * sin_val
    y1 = ey + eb * cos_val

    x2 = ex + eb * sin_val
    y2 = ey - eb * cos_val

    return (np.array([x1, y1]), np.array([x2, y2]))


def ellipse_point_cloud_squared_error(ex, ey, ea, eb, erad, xs, ys):
    """
    楕円と点群の距離の二乗誤差の合計を計算する。

    Parameters
    ----------
    ex, ey : float
        楕円の中心座標
    ea, eb : float
        楕円の長半径、短半径
    erad : float
        楕円の回転角 [rad]
    xs, ys : list or numpy.ndarray
        点群のX座標、Y座標

    Returns
    -------
    float
        各点から楕円までの最短距離の二乗和
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys):
        raise ValueError("xs と ys は同じ長さの一次元配列にしてください")
    if len(xs) == 0:
        raise ValueError("点群は1点以上必要です")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("点群に NaN または inf が含まれています")
    if not np.all(np.isfinite([ex, ey, ea, eb, erad])):
        raise ValueError("楕円パラメータに NaN または inf が含まれています")
    if ea <= 0 or eb <= 0:
        raise ValueError("ea と eb は正である必要があります")

    # 楕円を長半径 ea >= eb の形式にそろえる
    if ea < eb:
        ea, eb = eb, ea
        erad += np.pi / 2

    # 各点を楕円中心基準の座標に変換し、楕円の回転を戻す
    dx = xs - ex
    dy = ys - ey

    c = np.cos(erad)
    s = np.sin(erad)

    X = c * dx + s * dy
    Y = -s * dx + c * dy

    # 楕円周上の点 (ea*cos(t), eb*sin(t)) との距離二乗
    def dist2(t, px, py):
        return (ea * np.cos(t) - px) ** 2 + (eb * np.sin(t) - py) ** 2

    total = 0.0

    for px, py in zip(X, Y):
        # 全周を複数区間に分け、局所最小値を探索
        ts = np.linspace(0.0, 2.0 * np.pi, 65)
        vals = (ea * np.cos(ts) - px) ** 2 + (eb * np.sin(ts) - py) ** 2

        best = float(np.min(vals))

        # サンプル点での局所最小値の近傍を細かく探索
        for i in range(1, len(ts) - 1):
            if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
                res = minimize_scalar(dist2, args=(px, py), bounds=(ts[i - 1], ts[i + 1]), method="bounded", options={"xatol": 1e-12})
                if res.fun < best:
                    best = float(res.fun)

        total += best

    return float(total) / len(xs)


def detect_ellipse_2(binary_img, binary_img2, ratio=-1.0, flg_show=False):

    h, w = binary_img.shape
    w2 = w >> 1
    h2 = h >> 1

    ys, xs = np.where(0 < binary_img)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    xs -= w2
    ys = h2 - ys

    ys2, xs2 = np.where(0 < binary_img2)
    xs2 = np.array(xs2, dtype=np.float64)
    ys2 = np.array(ys2, dtype=np.float64)
    xs2 -= w2
    ys2 = h2 - ys2

    img_gt = debug_plot2(xs, ys, 1.0, w, h)
    img_gt2 = debug_plot2(xs2, ys2, 1.0, w, h)
    if flg_show:
        img_gt_or = img_gt | img_gt2

    # cv2.imshow("img_gt", img_gt)
    # cv2.imshow("img_gt2", img_gt2)
    # cv2.imshow("img_gt_or", img_gt_or)
    # cv2.waitKey(0)

    shrink_scale = 1000.0  # OK
    # shrink_scale = 2000.0 # OK
    # shrink_scale = g_scale

    xs /= shrink_scale
    ys /= shrink_scale
    xs2 /= shrink_scale
    ys2 /= shrink_scale

    xy = []
    for k, xx in enumerate(xs):
        yy = ys[k]
        xy.append((xx, yy))

    xy2 = []
    for k, xx in enumerate(xs2):
        yy = ys2[k]
        xy2.append((xx, yy))

    ex, ey, ea, eb, angle_rad = fit_ellipse(xs, ys)

    if flg_show:
        print("fit_ellipse_from_gray: (ex, ey, ea, eb, angle_rad)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (ex, ey, ea, eb, angle_rad))
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)
        draw_ellipse_to_screen_rgb(img_gt_rgb, shrink_scale, ex, ey, ea, eb, angle_rad, 2000, (0, 255, 0), 0.0, True)
        cv2.imshow("img_gt_rgb", img_gt_rgb)

    xpts = calc_ellipse_line_intersection(ex, ey, ea, eb, angle_rad)  # cross points
    # print(xpts[0], xpts[1])
    if xpts[1][1] < xpts[0][1]:
        xpt = xpts[1]
    else:
        xpt = xpts[0]
    # print("cross point lower:", xpt)

    lst_initial = []
    # lst_img_gt_rgb = []

    cpt = np.array([ex, ey])  # center point of the ellapse
    for k, s_pos in enumerate([0.8, 0.9, 1.0, 1.1, 1.2]):
        dv = s_pos * (cpt - xpt)
        c_new = dv + xpt  # new center of an ellipse
        eb_new = np.linalg.norm(dv)
        # print("(dv, c_new, eb_new):", dv, c_new, eb_new)
        # print("try: (ex, ey, ea, eb, angle_rad)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (c_new[0], c_new[1], ea, eb_new, angle_rad))
        _x, _y, _z, _ax, _az = ellipse_2d_to_3d(c_new[0], c_new[1], ea, eb_new, angle_rad)
        # print("ellipse_2d_to_3d: (_x, _y, _z, _ax, _az)=(%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (_x, _y, _z, _ax, _az))

        # img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)
        # project_3d_circle_to_screen_rgb(img_gt_rgb, shrink_scale, _x, _y, _z, _ax, 1.0, _az, g_rcoil, 1000, (0, 0, 255), 0.0, True)
        # lst_img_gt_rgb.append(img_gt_rgb)

        initial = (_x, _y, _z, _ax, _az)
        lst_initial.append(initial)

    x, y, z, ax, az, result = fit_circle_from_projected_points(xy, xy2, lst_initial, n_starts=1)

    if 0.0 < ratio:
        cx, cy, a, b, angle_rad = get_parameters_of_ellipse(x, y, z, ax, 1.0, az, 1.0)
        cx2, cy2, a2, b2, angle_rad2 = get_parameters_of_ellipse(x, y, z, ax, 1.0, az, ratio)
        # print("(cx, cy, a, b, angle_rad)      = (%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (cx, cy, a, b, angle_rad))
        # print("(cx2, cy2, a2, b2, angle_rad2) = (%8.6f, %8.6f, %8.6f, %8.6f, %8.6f)" % (cx2, cy2, a2, b2, angle_rad2))
        mse = ellipse_point_cloud_squared_error(cx, cy, a, b, angle_rad, xs, ys)
        mse2 = ellipse_point_cloud_squared_error(cx2, cy2, a2, b2, angle_rad2, xs2, ys2)
    else:
        mse = 1.0
        mse2 = 1.0

    if flg_show:
        img_gt_or_rgb = cv2.cvtColor(img_gt_or, cv2.COLOR_GRAY2RGB)
        project_3d_circle_to_screen_rgb(img_gt_or_rgb, shrink_scale, x, y, z, ax, 1.0, az, 1.0, 1200, (0, 255, 0), 0.0, True)
        project_3d_circle_to_screen_rgb(img_gt_or_rgb, shrink_scale, x, y, z, ax, 1.0, az, g_ratio, 600, (0, 0, 255))
        print("%5s, %d, %3d, %.6f, %.9f" % (result.success, result.status, result.nfev, result.cost, result.optimality))
        cv2.imshow("img_gt_or_rgb", img_gt_or_rgb)
        cv2.waitKey(10)

    return x, y, z, ax, az, result, mse, mse2


def _test_ellipse_fit3_2(x0, y0, z0, ax0, az0, ratio=-1.0, flg_show=False):
    wimg = 800
    himg = 800
    binary_img = np.zeros((himg, wimg), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img, g_scale, x0, y0, z0, ax0, 1.0, az0, g_rcoil, 1000, 1.7, False, True)
    binary_img2 = np.zeros((himg, wimg), dtype=np.uint8)
    project_3d_circle_to_screen(binary_img2, g_scale, x0, y0, z0, ax0, 1.0, az0, g_ratio * g_rcoil, 500, 1.7, False, True)
    # if flg_show:
    #    cv2.imshow("binary_img", binary_img)
    #    cv2.imshow("binary_img2", binary_img2)
    #    cv2.waitKey(0)
    return detect_ellipse_2(binary_img, binary_img2, ratio, flg_show)


def test_ellipse_fit4():
    binary_img = cv2.imread(g_path_img_ellipse_0, cv2.IMREAD_GRAYSCALE)
    binary_img2 = cv2.imread(g_path_img_ellipse_1, cv2.IMREAD_GRAYSCALE)
    detect_ellipse_2(binary_img, binary_img2, g_ratio, True)
    cv2.waitKey(0)


def test_ellipse_fit3():
    # _func = _test_ellipse_fit3
    _func = _test_ellipse_fit3_2
    idx = 0
    ddx = 0.2
    ddy = 0.2
    ddz = 0.2
    ddax = 0.1
    ddaz = 0.1
    for dx in [-ddx, 0.0, ddx]:
        for dy in [-ddy, 0.0, ddy]:
            for dz in [-ddz, 0.0, ddz]:
                for dax in [-ddax, 0.0, ddax]:
                    for daz in [-ddaz, 0.0, ddaz]:
                        # print("------------------------------------------------------------- %6d" % idx)
                        x, y, z, ax, az, result, mse, mse2 = _func(g_x + dx, g_y + dy, g_z + dz, g_ax + dax, g_az + daz, g_ratio)
                        print("%12.9f, %12.9f" % (np.sqrt(mse), np.sqrt(mse2)))
                        idx += 1
                        # exit()


if __name__ == "__main__":
    # test_residuals()
    # test_residuals2()
    # test_residuals3()
    # test_residuals4()
    # test_ellipse_fit()
    # test_ellipse_fit2()
    test_ellipse_fit3()
    # test_ellipse_fit4()
