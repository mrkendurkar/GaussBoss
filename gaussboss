import numpy as np
import subprocess
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import get_body_barycentric
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# CONFIG
# -----------------------------
images = [
    ".fits",
    ".fits",
    ".fits",
]

#you can add as many images you like

tol = 2.0
sex_config = "auto.sex"

# -----------------------------
# SExtractor
# -----------------------------
def run_sextractor(fits_file, cat_file):
    subprocess.run([
        "source-extractor",
        "-c", sex_config,
        fits_file,
        "-CATALOG_NAME", cat_file
    ], check=True)

# -----------------------------
# Read XY + WCS + Time
# -----------------------------
def get_data(img, cat):
    data = ascii.read(cat)

    # auto-detect X/Y columns
    x_col = next((c for c in data.colnames if "X" in c.upper()), None)
    y_col = next((c for c in data.colnames if "Y" in c.upper()), None)

    if x_col is None or y_col is None:
        raise ValueError(f"No X/Y columns in {cat}")

    xy = np.column_stack((data[x_col], data[y_col]))

    header = fits.getheader(img)
    wcs = WCS(header)

    # observation time
    if "JD" in header:
        t = Time(header["JD"], format="jd")
    elif "MJD" in header:
        t = Time(header["MJD"], format="mjd")
    else:
        t = Time(header["DATE-OBS"])

    return xy, wcs, t

# -----------------------------
# Build catalogs
# -----------------------------
xy_list, wcs_list, time_list = [], [], []

for i, img in enumerate(images):
    cat = f"cat{i+1}.cat"
    run_sextractor(img, cat)

    xy, wcs, t = get_data(img, cat)
    xy_list.append(xy)
    wcs_list.append(wcs)
    time_list.append(t)

# -----------------------------
# Track linking
# -----------------------------
def build_tracks(xy_list, tol):
    tracks = []
    ref = xy_list[0]

    for x0, y0 in ref:
        track = [(0, x0, y0)]
        last_x, last_y = x0, y0

        for i in range(1, len(xy_list)):
            xy = xy_list[i]

            dx = np.abs(xy[:,0] - last_x)
            dy = np.abs(xy[:,1] - last_y)

            idx = np.where((dx <= tol) & (dy <= tol))[0]
            if len(idx) == 0:
                break

            j = idx[0]
            last_x, last_y = xy[j]
            track.append((i, last_x, last_y))

        if len(track) >= 3:
            tracks.append(track)

    return tracks

tracks = build_tracks(xy_list, tol)

# -----------------------------
# Linear filter
# -----------------------------
def is_linear(track):
    xs = np.array([p[1] for p in track])
    ys = np.array([p[2] for p in track])

    return np.std(np.diff(xs)) < 1.5 and np.std(np.diff(ys)) < 1.5

tracks = [t for t in tracks if is_linear(t)]

# -----------------------------
# Convert to RA/Dec
# -----------------------------
def to_radec(track):
    out = []
    for i, x, y in track:
        ra, dec = wcs_list[i].all_pix2world(x, y, 1)
        out.append((time_list[i], ra, dec))
    return out

radec_tracks = [to_radec(t) for t in tracks]

# -----------------------------
# Velocity
# -----------------------------
def compute_velocity(track):
    ras = np.array([p[1] for p in track])
    decs = np.array([p[2] for p in track])
    times = np.array([p[0].jd for p in track])

    dra = (ras - ras[0]) * 3600 * np.cos(np.deg2rad(decs))
    ddec = (decs - decs[0]) * 3600
    dt = (times - times[0]) * 24

    if dt[-1] == 0:
        return 0

    return np.sqrt(dra[-1]**2 + ddec[-1]**2) / dt[-1]

# -----------------------------
# ML filtering
# -----------------------------
def extract_features(track):
    xs = np.array([p[1] for p in track])
    ys = np.array([p[2] for p in track])

    return [
        len(track),
        np.std(np.diff(xs)),
        np.std(np.diff(ys)),
        compute_velocity(track)
    ]

velocities = [compute_velocity(t) for t in radec_tracks]

# physical filter
filtered = [t for t, v in zip(radec_tracks, velocities) if 1 < v < 1000]

print(f"Tracks before velocity filter: {len(radec_tracks)}")
print(f"Tracks after velocity filter: {len(filtered)}")

# If nothing passes → fallback
if len(filtered) == 0:
    print("⚠️ No tracks passed velocity filter — using all tracks")
    filtered = radec_tracks

# ML only if enough samples
if len(filtered) >= 5:
    X = np.array([extract_features(t) for t in filtered])

    model = IsolationForest(contamination=0.3, random_state=42)
    labels = model.fit_predict(X)

    final_tracks = [t for t, l in zip(filtered, labels) if l == 1]
else:
    print("⚠️ Too few tracks for ML — skipping ML filter")
    final_tracks = filtered

# -----------------------------
# ORBIT DETERMINATION
# -----------------------------
def radec_to_unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.array([
        np.cos(dec)*np.cos(ra),
        np.cos(dec)*np.sin(ra),
        np.sin(dec)
    ])

def earth_pos(t):
    p = get_body_barycentric("earth", t)
    return np.array([p.x.value, p.y.value, p.z.value])

def preliminary_orbit(track):
    t1, ra1, dec1 = track[0]
    t2, ra2, dec2 = track[len(track)//2]
    t3, ra3, dec3 = track[-1]

    rho1 = radec_to_unit(ra1, dec1)
    rho2 = radec_to_unit(ra2, dec2)
    rho3 = radec_to_unit(ra3, dec3)

    R1, R2, R3 = earth_pos(t1), earth_pos(t2), earth_pos(t3)

    r2 = 1.0  # rough guess (AU)

    r_vec = r2 * rho2 - R2
    r_vec1 = r2 * rho1 - R1
    r_vec3 = r2 * rho3 - R3

    dt = (t3.jd - t1.jd) * 86400
    v_vec = (r_vec3 - r_vec1) / dt

    return r_vec, v_vec

def orbital_elements(r, v):
    AU = 1.496e8
    mu = 1.32712440018e11

    r, v = r*AU, v*AU

    h = np.cross(r, v)
    i = np.degrees(np.arccos(h[2]/np.linalg.norm(h)))

    e_vec = (np.cross(v, h)/mu) - r/np.linalg.norm(r)
    e = np.linalg.norm(e_vec)

    a = 1 / ((2/np.linalg.norm(r)) - (np.linalg.norm(v)**2/mu))

    return a/AU, e, i

# -----------------------------
# MPC OUTPUT
# -----------------------------
def mpc_line(obj, t, ra, dec):
    t = t.utc
    date = t.strftime("%Y %m %d.%f")[:16]

    ra_h = ra/15
    h = int(ra_h)
    m = int((ra_h-h)*60)
    s = (ra_h-h-m/60)*3600

    d = int(dec)
    dm = int(abs(dec-d)*60)
    ds = abs(dec-d-dm/60)*3600

    return f"{obj:5s} C{date} {h:02d} {m:02d} {s:05.2f} {d:+03d} {dm:02d} {ds:04.1f}"

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("mpc_report.txt", "w") as f:
    for i, track in enumerate(final_tracks):
        obj = f"T{i:04d}"
        vel = compute_velocity(track)

        r, v = preliminary_orbit(track)
        a, e, inc = orbital_elements(r, v)

        print(f"\nObject {obj}")
        print(f"Velocity: {vel:.1f} arcsec/hr")
        print(f"a={a:.2f} AU, e={e:.2f}, i={inc:.2f} deg")

        for t, ra, dec in track:
            f.write(mpc_line(obj, t, ra, dec) + "\n")

print("\n✅ DONE: MPC report + orbit estimation complete")

# -----------------------------
# MPC FORMAT 
# -----------------------------
def format_ra(ra):
    ra_h = ra / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = (ra_h - h - m/60) * 3600
    return f"{h:02d} {m:02d} {s:05.2f}"

def format_dec(dec):
    sign = "+" if dec >= 0 else "-"
    dec = abs(dec)
    d = int(dec)
    m = int((dec - d) * 60)
    s = (dec - d - m/60) * 3600
    return f"{sign}{d:02d} {m:02d} {s:04.1f}"

def format_date(t):
    t = t.utc
    year = t.datetime.year
    month = t.datetime.month
    day = t.datetime.day + (t.datetime.hour +
                           t.datetime.minute/60 +
                           t.datetime.second/3600) / 24
    return f"{year:4d} {month:02d} {day:08.5f}"

# -----------------------------
# SAVE MPC FILE
# -----------------------------
output_file = "mpc_report.txt"

with open(output_file, "w") as f:
    for i, track in enumerate(final_tracks):

        obj_id = f"T{i:04d}"   # temporary designation
        vel = compute_velocity(track)

        for t, ra, dec in track:
            date_str = format_date(t)
            ra_str = format_ra(ra)
            dec_str = format_dec(dec)

            # MPC 80-column format (basic)
            line = f"{obj_id:5s} C{date_str} {ra_str} {dec_str}          "

            f.write(line + "\n")

        # optional: add velocity comment line
        f.write(f"# Velocity: {vel:.2f} arcsec/hr\n\n")

print(f"✅ MPC file saved as: {output_file}")
