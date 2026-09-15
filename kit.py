#!/usr/bin/env python
"""
kit.py -- photos in, submission out.

    pip install numpy pillow pyproj scipy      # plus pillow-heif for HEIC photos

    python kit.py observed my_photos/          # which grid cells your photos saw
    python kit.py check teamA_submission.npy   # shape and class histogram

    from kit import *
    grid, photos = load("my_photos/")
    sub = empty()
    for p in photos:
        paint_cone(sub, p, BROADLEAF)          # 60 deg / 150 m cone, 50 m disk without a heading
    fill_rest(sub, OPEN)
    save(sub, "teamA_submission.npy")

The grid comes from meta.json beside this file. What a photo shows, and what belongs in a cell
no photo reached, is yours to decide.
"""
import csv
import json
import math
import os
import sys

import numpy as np

try:
    import pillow_heif                        # lets PIL open HEIC
    pillow_heif.register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "meta.json")))
UNKNOWN, BROADLEAF, CONIFER, OPEN, WATER, BUILT = 0, 1, 2, 3, 4, 5
NAMES = {int(k): v for k, v in META["classes"].items()}
PALETTE = {int(k): v for k, v in META["palette_rgb"].items()}
ROWS, COLS = META["grid_shape"]
FOV, DEPTH, DISK = 60.0, 150.0, 50.0          # the convention we score "observed" with
EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------- EXIF
def _deg(dms, ref):
    d, m, s = (x[0] / x[1] if isinstance(x, tuple) else float(x) for x in dms)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif(path):
    """lat, lon, heading, datetime -- None for whatever the photo does not carry."""
    from PIL import Image
    with Image.open(path) as im:
        exif = im.getexif()
        stamp = exif.get_ifd(0x8769).get(0x9003) or exif.get(306)
        gps = exif.get_ifd(0x8825)
    if not gps or 2 not in gps or 4 not in gps:
        return {"lat": None, "lon": None, "heading": None, "datetime": stamp}
    h = gps.get(17)
    return {"lat": _deg(gps[2], gps.get(1)), "lon": _deg(gps[4], gps.get(3)),
            "heading": float(h) % 360 if h is not None else None, "datetime": stamp}


# ---------------------------------------------------------------- grid
class Grid:
    def __init__(self, meta=META):
        from pyproj import Transformer
        self.rows, self.cols = meta["grid_shape"]
        self.cell = float(meta["cell_size_m"])
        self.xmin, self.ymax = meta["origin_top_left_2154"]
        self._to_m = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        cx = self.xmin + (np.arange(self.cols) + 0.5) * self.cell
        cy = self.ymax - (np.arange(self.rows) + 0.5) * self.cell
        self.CX, self.CY = np.meshgrid(cx, cy)

    def latlon_to_xy(self, lat, lon):
        return self._to_m.transform(lon, lat)

    def xy_to_rc(self, x, y):
        return (int((self.ymax - y) // self.cell), int((x - self.xmin) // self.cell))

    def inside(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def frustum(self, x, y, heading, fov=FOV, depth=DEPTH):
        dx, dy = self.CX - x, self.CY - y
        bearing = np.degrees(np.arctan2(dx, dy)) % 360          # 0 = north, clockwise
        return (np.hypot(dx, dy) <= depth) & (np.abs((bearing - heading + 180) % 360 - 180) <= fov / 2)

    def disk(self, x, y, radius=DISK):
        return np.hypot(self.CX - x, self.CY - y) <= radius


GRID = Grid()


# ---------------------------------------------------------------- photos
def load(photo_dir, csv_out=None):
    """Every image under photo_dir with usable EXIF GPS, sorted by capture time."""
    photos = []
    for root, _, names in os.walk(photo_dir):
        for name in sorted(names):
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in EXTS:
                continue
            e = read_exif(os.path.join(root, name))
            if e["lat"] is None:
                print(f"  no EXIF GPS, skipped: {name}")
                continue
            x, y = GRID.latlon_to_xy(e["lat"], e["lon"])
            r, c = GRID.xy_to_rc(x, y)
            photos.append({"file": os.path.join(root, name), "name": name, "x": x, "y": y,
                           "row": r, "col": c, "inside": GRID.inside(r, c), **e})
    if not photos:
        raise SystemExit(f"No photo under {photo_dir} carries EXIF GPS.\n"
                         "Location has to be on when the photo is taken, and you have to copy the\n"
                         "originals: chat apps and 'optimised' cloud copies strip the GPS out.")
    outside = [p for p in photos if not p["inside"]]
    if outside:
        print(f"  {len(outside)} photo(s) fall outside the grid and paint nothing: "
              + ", ".join(p["name"] for p in outside[:3]) + ("..." if len(outside) > 3 else ""))
    photos.sort(key=lambda p: (p["datetime"] or "", p["name"]))
    if csv_out:
        keys = [k for k in photos[0] if k != "file"]
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: p[k] for k in keys} for p in photos])
    return GRID, photos


def seen(photo, fov=FOV, depth=DEPTH, disk=DISK):
    """The cells a photo sees, by the same rule we score with."""
    if photo["heading"] is None:
        m = GRID.disk(photo["x"], photo["y"], disk)
    else:
        m = GRID.frustum(photo["x"], photo["y"], photo["heading"], fov, depth)
    if photo["inside"]:
        m[photo["row"], photo["col"]] = True
    return m


# ---------------------------------------------------------------- painting
def empty():
    return np.zeros((ROWS, COLS), np.uint8)


def paint_cell(sub, row, col, cls):
    if GRID.inside(row, col):                 # negative indices would wrap to the far edge
        sub[row, col] = cls


def paint_cone(sub, photo, cls, **kw):
    sub[seen(photo, **kw)] = cls


def paint_rect(sub, row0, row1, col0, col1, cls):
    """Rows row0..row1 and columns col0..col1, inclusive."""
    sub[row0:row1 + 1, col0:col1 + 1] = cls


def fill_rest(sub, cls):
    sub[sub == 0] = cls


def nearest_fill(sub):
    """Every empty cell takes the class of the nearest painted one."""
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(sub == 0, return_distances=False, return_indices=True)
    sub[:] = sub[idx[0], idx[1]]
    return sub


# ---------------------------------------------------------------- output
def _rgb(grid):
    pal = np.zeros((256, 3), np.uint8)
    for k, v in PALETTE.items():
        pal[k] = v
    return pal[grid]


def show(sub, path="preview.png", scale=6):
    """An upscaled colour preview. Not a submission -- it is bigger than the grid."""
    from PIL import Image
    Image.fromarray(_rgb(sub)).resize((COLS * scale, ROWS * scale), Image.NEAREST).save(path)
    return path


def check(path):
    """Print the shape and class histogram of a submission."""
    sub = np.load(path)
    print(f"{path}: shape {sub.shape} (want ({ROWS}, {COLS})), values {sub.min()}..{sub.max()}")
    for k in range(6):
        n = int((sub == k).sum())
        print(f"  {k} {NAMES[k]:8s} {n:6d}  {100 * n / sub.size:5.1f}%")
    if (sub == 0).any():
        print("  unknown (0) is scored as wrong -- fill every cell")
    start = META["route_endpoints"]["start"]
    print(f"  row 0 is NORTH, col 0 is WEST; the campsite is at row {start['row']}, col {start['col']}")
    return sub


def save(sub, path):
    np.save(path, np.asarray(sub, np.uint8))
    check(path)


# ---------------------------------------------------------------- observed cells
def observed(photo_dir, out_dir="observed", limit=None):
    """Paint every photo's view into the grid; write the mask, the counts, a PNG and a CSV."""
    os.makedirs(out_dir, exist_ok=True)
    _, photos = load(photo_dir, csv_out=os.path.join(out_dir, "photos.csv"))
    photos = photos[:limit]
    count = np.zeros((ROWS, COLS), np.int32)
    for p in photos:
        count += seen(p)
    mask = count > 0
    np.save(os.path.join(out_dir, "observed_mask.npy"), mask)
    np.save(os.path.join(out_dir, "photo_count.npy"), count)
    from PIL import Image
    img = np.full((ROWS, COLS, 3), 225, np.uint8)
    t = (count / max(count.max(), 1))[..., None]
    img[mask] = (np.array([200, 225, 240]) * (1 - t) + np.array([15, 60, 105]) * t)[mask]
    Image.fromarray(img).resize((COLS * 5, ROWS * 5), Image.NEAREST).save(os.path.join(out_dir, "observed_mask.png"))
    heads = sum(p["heading"] is not None for p in photos)
    print(f"{len(photos)} photos ({heads} with a heading) -> {int(mask.sum())} of {mask.size} cells "
          f"({100 * mask.mean():.1f}%), written to {out_dir}/")
    return mask, count


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("observed", "check"):
        print(__doc__)
        sys.exit(0)
    cmd, arg = sys.argv[1], sys.argv[2]
    check(arg) if cmd == "check" else observed(arg, *sys.argv[3:4])
