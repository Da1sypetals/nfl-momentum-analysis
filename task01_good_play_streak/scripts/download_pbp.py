import concurrent.futures
import gzip
import pathlib
import urllib.request

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SEASONS = list(range(1999, 2027))


def download(season):
    path = RAW / f"play_by_play_{season}.csv.gz"
    if path.exists():
        with gzip.open(path, "rb") as f:
            f.read(1024)
        return season, path.stat().st_size, "exists"

    url = BASE.format(season=season)
    part = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, part)
    with gzip.open(part, "rb") as f:
        f.read(1024)
    part.rename(path)
    return season, path.stat().st_size, "downloaded"


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for season, size, status in pool.map(download, SEASONS):
        print(season, size, status, flush=True)
