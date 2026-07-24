"""Edge-entropy region selection + region-targeted corruption for PostVRG.

cv2/scipy-FREE (numpy + PIL only) so it runs in the OmniCap env used to run
postvrg.py. Ported/adapted from GPSToken/validate_adaptive_gps.py, with two
important changes for our use:
  * SELECTION uses edge DENSITY (entropy, no area term) so small detailed
    regions are picked -- the area-weighted complexity would pick large smooth
    regions (wrong direction). Splitting still uses the area-weighted score.
  * corruption (noise/blur) is applied in PIXEL space, feathered, to the
    selected detail regions only -- smooth regions stay identical to the
    conditional branch, so VCD's (cond - weak) contrast targets the detail.
"""
import numpy as np
from PIL import Image, ImageFilter


# ----------------------------------------------------------------------------- basics
def image_to_gray_norm(pil_img):
    """PIL RGB -> grayscale float array in [-1, 1], native resolution."""
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray / 255.0 * 2.0 - 1.0


def _resize64(img_norm):
    """Resize a [-1,1] grayscale region to 64x64 [0,1] using PIL (no cv2)."""
    a = (((img_norm + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)
    if a.shape[0] < 1 or a.shape[1] < 1:
        a = np.zeros((1, 1), np.uint8)
    im = Image.fromarray(a).resize((64, 64), Image.BILINEAR)
    return np.asarray(im).astype(np.float64) / 255.0


def _edge_entropy(img01_64):
    """Shannon entropy of the edge-magnitude histogram (detail density)."""
    gy, gx = np.gradient(img01_64)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hist, _ = np.histogram(mag, bins=512, range=(-5.5, 5.5))
    hist = hist / (hist.sum() + 1e-5)
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def calculate_complexity_degree(img_norm, weight=1.0):
    """Area-WEIGHTED complexity used to decide what to split (favours big regions)."""
    h, w = img_norm.shape
    ent = _edge_entropy(_resize64(img_norm))
    return ent ** weight * h * w


def region_edge_density(img_norm):
    """Area-INDEPENDENT detail score used to SELECT highlight regions."""
    return _edge_entropy(_resize64(img_norm))


# ----------------------------------------------------------------------------- split
def adaptive_initialize(gps_num, img_norm, minhw=8, weight=1.0):
    """Recursively split the image into `gps_num` regions, prioritising the
    highest area-weighted complexity. Returns region boxes in normalized [-1,1]."""
    h, w = img_norm.shape
    region_candidate = [(img_norm, calculate_complexity_degree(img_norm, weight), (0, h - 1), (0, w - 1))]

    while len(region_candidate) < gps_num:
        region_candidate.sort(key=lambda x: x[1], reverse=True)
        pick_one = None
        for _i in range(len(region_candidate)):
            _, _, _hr, _wr = region_candidate[_i]
            if _hr[1] - _hr[0] <= minhw and _wr[1] - _wr[0] <= minhw:
                continue
            pick_one = region_candidate.pop(_i)
            break
        if pick_one is None:
            break  # nothing left large enough to split

        _img, _, _hr, _wr = pick_one
        _h, _w = _img.shape
        splits = []
        if _w > minhw and _w >= _h:          # vertical split
            splits = [
                (_img[:, :_w // 2], _hr, (_wr[0], _wr[0] + _w // 2 - 1)),
                (_img[:, _w // 2:], _hr, (_wr[0] + _w // 2, _wr[1])),
            ]
        else:                                 # horizontal split
            splits = [
                (_img[:_h // 2, :], (_hr[0], _hr[0] + _h // 2 - 1), _wr),
                (_img[_h // 2:, :], (_hr[0] + _h // 2, _hr[1]), _wr),
            ]
        for sub, hr, wr in splits:
            region_candidate.append((sub, calculate_complexity_degree(sub, weight), hr, wr))

    regions = []
    for _, _, hr, wr in region_candidate:
        regions.append((
            (wr[0] / (w - 1) - 0.5) * 2, (hr[0] / (h - 1) - 0.5) * 2,
            (wr[1] / (w - 1) - 0.5) * 2, (hr[1] / (h - 1) - 0.5) * 2,
        ))
    return np.array(regions, dtype=np.float32)


# ----------------------------------------------------------------------------- select
def select_highlight_regions(img_norm, gps_num=16, minhw=8, weight=2.5,
                             quantile=0.5, abs_threshold=None, min_highlight=1):
    """Divide into regions, score each by edge density, return the detailed ones.

    Returns: regions_px [(x1,y1,x2,y2) pixels], densities, highlight_bool, threshold.
    """
    h, w = img_norm.shape
    regions = adaptive_initialize(gps_num, img_norm, minhw=minhw, weight=weight)

    regions_px, densities = [], []
    for (nx1, ny1, nx2, ny2) in regions:
        px1 = int((nx1 / 2.0 + 0.5) * (w - 1)); py1 = int((ny1 / 2.0 + 0.5) * (h - 1))
        px2 = int((nx2 / 2.0 + 0.5) * (w - 1)); py2 = int((ny2 / 2.0 + 0.5) * (h - 1))
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)
        sub = img_norm[py1:py2 + 1, px1:px2 + 1]
        if sub.size == 0:
            sub = img_norm[py1:py1 + 1, px1:px1 + 1]
        densities.append(region_edge_density(sub))
        regions_px.append((px1, py1, px2, py2))

    densities = np.array(densities)
    thr = abs_threshold if abs_threshold is not None else float(np.quantile(densities, quantile))
    highlight = densities > thr
    if highlight.sum() < min_highlight:      # fallback: force the top-k densest
        order = np.argsort(-densities)[:min_highlight]
        highlight = np.zeros_like(highlight); highlight[order] = True
    return regions_px, densities, highlight, thr


# ----------------------------------------------------------------------------- corrupt
def build_soft_mask(hw, regions_px, feather_px):
    """Feathered [0,1] mask (1 inside highlight regions), Gaussian-softened border."""
    h, w = hw
    m = np.zeros((h, w), np.float32)
    for (x1, y1, x2, y2) in regions_px:
        m[y1:y2 + 1, x1:x2 + 1] = 1.0
    if feather_px and feather_px > 0:
        im = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(feather_px)))
        m = np.asarray(im).astype(np.float32) / 255.0
    return m


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _diffusion_noise_pixels(img01, noise_step, seed=None):
    """Forward-diffusion corruption in pixel space (same schedule as the model's)."""
    betas = _sigmoid(np.linspace(-6, 6, 1000)) * (0.005 - 1e-5) + 1e-5
    alpha_bar = float(np.cumprod(1.0 - betas)[int(noise_step)])
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(img01.shape)
    return np.sqrt(alpha_bar) * img01 + np.sqrt(1.0 - alpha_bar) * noise


def random_box(id_str, W, H, crop_frac):
    """One random box of size crop_frac*(W,H) at a random position, seeded
    deterministically by the sample id -> the SAME box every call. Used so the
    draft (box noised) and the fill-in (box cropped) target the identical region."""
    import zlib, random as _random
    cw = max(1, int(crop_frac * W)); ch = max(1, int(crop_frac * H))
    rng = _random.Random(zlib.crc32(str(id_str).encode()) & 0xFFFFFFFF)
    x1 = rng.randint(0, W - cw) if W - cw > 0 else 0
    y1 = rng.randint(0, H - ch) if H - ch > 0 else 0
    return (x1, y1, x1 + cw - 1, y1 + ch - 1)


def apply_box_noise(pil_img, box, noise_step=200, seed=42, feather_frac=0.02):
    """Diffusion-noise a single rectangular box region (feathered), leave the rest
    clean. The random-selection analogue of apply_region_corruption(edge_noise)."""
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float64)
    H, W = rgb.shape[:2]
    feather_px = max(1, int(feather_frac * W))
    soft = build_soft_mask((H, W), [box], feather_px)[..., None]
    img01 = rgb / 255.0
    noise = _diffusion_noise_pixels(img01, noise_step=noise_step, seed=seed)
    out = img01 * (1.0 - soft) + noise * soft
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))


def apply_region_corruption(pil_img, gps_num=16, quantile=0.5,
                            weight=2.5, minhw=8, feather_frac=0.02,
                            noise_step=200, seed=42, min_highlight=4,
                            return_debug=False, regions=None):
    """Diffusion-noise ONLY the edge-dense (detail) regions of the image, feathered;
    leave the smooth regions clean. Used for the draft edge_noise mode so the draft
    focuses on global/smooth semantics. Same size -> same anyres token count.

    `regions` = a precomputed (regions_px, densities, hi, thr) tuple from
    select_highlight_regions; pass it to reuse ONE selection across stages."""
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float64)
    H, W = rgb.shape[:2]

    if regions is not None:
        regions_px, dens, hi, thr = regions
    else:
        regions_px, dens, hi, thr = select_highlight_regions(
            image_to_gray_norm(pil_img), gps_num=gps_num, minhw=minhw, weight=weight,
            quantile=quantile, min_highlight=min_highlight)
    hi_regions = [regions_px[i] for i in range(len(regions_px)) if hi[i]]

    feather_px = max(1, int(feather_frac * W))
    soft = build_soft_mask((H, W), hi_regions, feather_px)[..., None]  # [H,W,1]

    img01 = rgb / 255.0
    corrupt = _diffusion_noise_pixels(img01, noise_step=noise_step, seed=seed)
    weak01 = img01 * (1.0 - soft) + corrupt * soft
    weak_img = Image.fromarray((weak01.clip(0, 1) * 255).astype(np.uint8))

    if return_debug:
        return weak_img, dict(regions_px=regions_px, densities=dens, highlight=hi,
                              threshold=thr, n_highlight=int(hi.sum()))
    return weak_img


def apply_region_spotlight(pil_img, gps_num=16, quantile=0.5, weight=2.5, minhw=8,
                           feather_frac=0.02, noise_step=200, seed=42,
                           min_highlight=4, return_debug=False, regions=None):
    """Keep the whole image but SPOTLIGHT the edge-dense (detail) regions: those
    regions stay clean while the background is replaced with diffusion noise.
    Feathered; same size -> same anyres token count; preserves global context
    (unlike crop). The 'reverse' of the edge_noise draft.

    `regions` = a precomputed (regions_px, densities, hi, thr) tuple from
    select_highlight_regions; pass it to reuse ONE selection across stages."""
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float64)
    H, W = rgb.shape[:2]

    if regions is not None:
        regions_px, dens, hi, thr = regions
    else:
        regions_px, dens, hi, thr = select_highlight_regions(
            image_to_gray_norm(pil_img), gps_num=gps_num, minhw=minhw, weight=weight,
            quantile=quantile, min_highlight=min_highlight)
    hi_regions = [regions_px[i] for i in range(len(regions_px)) if hi[i]]

    feather_px = max(1, int(feather_frac * W))
    soft = build_soft_mask((H, W), hi_regions, feather_px)[..., None]  # 1 in highlight
    img01 = rgb / 255.0
    bg = _diffusion_noise_pixels(img01, noise_step=noise_step, seed=seed)
    out01 = img01 * soft + bg * (1.0 - soft)                           # detail clean, bg noised
    spot_img = Image.fromarray((out01.clip(0, 1) * 255).astype(np.uint8))

    if return_debug:
        return spot_img, dict(regions_px=regions_px, densities=dens, highlight=hi,
                              threshold=thr, n_highlight=int(hi.sum()))
    return spot_img
