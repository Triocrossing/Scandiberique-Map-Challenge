# Scandibérique Map Challenge

You ride ~3 km of the Scandibérique towpath from Camping La Rivière Dorée (Bagneaux-sur-Loing) north to
the Château de Nemours, photographing as you go. Then you rebuild a **bird's-eye land-cover map** of the
5 × 6 km rectangle around the route, from at most **30 of your own photos** plus an LLM used as a local
guide.

Most of the map is cells no photo ever saw. Filling those is the point, and it is scored separately.
Everything you need is `slides.pdf` (the briefing), this file, `meta.json` and `kit.py`.

## The grid

| | |
|---|---|
| shape | **121 rows × 101 columns**, in `meta.json` as `grid_shape` |
| cell | 50 m, about eight seconds of riding |
| orientation | **row 0 is the north edge, column 0 is the west edge** |
| landmarks | campsite at row 87, col 56 · château at row 33, col 44 |
| CRS | EPSG:2154 (Lambert-93), metres |

A point `(x, y)` in EPSG:2154 lands in `col = floor((x - xmin) / 50)` and `row = floor((ymax - y) / 50)`,
with `xmin, ymax = meta["origin_top_left_2154"]`. From GPS, project first:
`Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform(lon, lat)`. Both are already
written for you — `Grid.latlon_to_xy` and `Grid.xy_to_rc` in `kit.py`.

## The five classes

| id | class | what counts |
|---|---|---|
| 1 | broadleaf | woods of wide-leaved trees — oak, beech, hornbeam, birch, chestnut. A closed leafy canopy, pale green in summer, trunks that branch |
| 2 | conifer | needle-leaved woods, here almost entirely Scots pine. Dark, straight bare trunks, often in planted rows, a floor of needles rather than leaves |
| 3 | open | no tree cover, not water, not built: crops, stubble, meadow, pasture, orchard, allotments, cemetery, sports pitch, park lawn, scrub |
| 4 | water | the Loing, the Canal du Loing, lakes, gravel-pit ponds. Not a stream you can step over |
| 5 | built | whole residential areas — houses *with* their gardens — plus industry, retail, car parks, sports centres, rail land |
| 0 | unknown | scored as wrong. Fill every cell; guess rather than leave it blank |

One class per cell. Where things overlap the priority is **water › built › forest › open**, applied to
whatever covers at least a quarter of the cell; within forest, the leaf type that covers more wins. So a
road under ~8 m wide does not make a cell built: a country road through pines stays conifer, a village
street becomes built.

**There is no "mixed" class.** Stands that are genuinely mixed, and stands the source does not resolve,
take the leaf type of the nearest stand that is resolved — about a fifth of the grid is decided that way.
So do not agonise over an oak wood with a few pines in it; get the obvious ones right.

From the saddle: a wall of wide leaves = broadleaf. Dark straight trunks and a needle floor = conifer. A
view across fields = open. The canal or the river beside you = water. Houses, walls, fences = built.

## Rules

**Allowed** — any program you write (CLIP, any vision model, colour heuristics, geometry, rules), and any
LLM, online or offline, **with web search and browsing switched off**. Its memory is your prior: ask it what
the Loing valley looks like, which side the forest is on, how wide a canal is.

**Not allowed** — painting cells by hand, since the map must come out of your program; anything that shows
or describes a map (Apple Maps, Google Maps, OpenStreetMap, satellite imagery, Wikipedia, search engines);
an LLM with search on; photos not taken on the ride, or more than 30 of them.

## Three sub-problems worth separating

1. **Recognise** — what land cover is in each photo? Zero-shot CLIP, a small VLM, colour statistics, or the
   LLM reading your captions. Broadleaf against conifer is the sharpest call a photo can make and also the
   easiest to get wrong at a distance. Photos disagree with each other; decide how to combine them.
2. **Project** — where on the grid does that belong? A photo has a position and maybe a heading. What it
   shows is in front of the camera, at some depth, on one side of the path. Foreground and background land
   in different cells; the river in the frame continues for kilometres.
3. **Densify** — 30 photos reach a few percent of 12 221 cells. Rivers are lines, towns are blobs around a
   centre, fields and forest come in patches. Propagate, smooth, fit shapes. This is the unobserved score.

## What you hand in

* `<team>_submission.npy` — a uint8 array of shape (121, 101), values 1–5.
* `<team>_photos/` — the at most 30 originals with EXIF intact. We compute your observed cells from these.

## A first map in fifteen minutes

```bash
pip install numpy pillow pyproj scipy      # plus pillow-heif if your photos are HEIC
```

```python
from kit import *

grid, photos = load("my_photos/")     # EXIF GPS + heading -> grid cells
sub = empty()                         # 121 x 101, all zeros
for p in photos:
    cls = OPEN                        # <- per photo: BROADLEAF / CONIFER / OPEN / WATER / BUILT
    paint_cone(sub, p, cls)           # 60 deg, 150 m cone; 50 m disk if the photo has no heading
fill_rest(sub, BROADLEAF)             # <- your prior for every cell no photo reached
save(sub, "teamA_submission.npy")     # writes it and prints the class histogram
```

That scores about 0.07 mean IoU, around the same as painting the single commonest class everywhere.
Everything above that is yours.
`show(sub)` writes a colour preview; `nearest_fill(sub)` gives every empty cell its nearest painted
neighbour, which is a better floor than `fill_rest`.

## Which cells did your photos see?

```bash
python kit.py observed my_photos/
```

Writes a boolean mask, a per-cell photo count, a PNG and a CSV of every parsed photo position. We use the
same function with the same parameters — a 60° cone 150 m deep along the EXIF heading, or a 50 m disk when
the photo carries no heading — so this tells you exactly which cells count as observed for you.

## Scoring

Mean IoU over the five classes, on all cells, on the cells your photos saw, and on the cells neither team
saw. IoU is averaged per class, so water counts as much as broadleaf even though there is far less of it.
Two rankings: **map** (all cells) and **prior** (the cells nobody saw). The floor, painting the commonest
class everywhere, is about 0.07.

We also report the score with the two leaf types merged into one forest class. The gap between that and
your five-class score is exactly what broadleaf-versus-conifer cost you, which separates a shaky eye for
trees from a shaky map.

## Before you ride

**Stop the bike before you take a photo.** It is a towpath with other people on it, and nothing here is
worth a crash. Keep location on: the EXIF GPS tag is the only thing that places a photo on the grid, and a
photo without it is dead weight. Send originals — chat apps and "optimised" copies strip it out. The 30 is
per team, not per person, so agree on the bike who covers what, and shoot more than you need.
