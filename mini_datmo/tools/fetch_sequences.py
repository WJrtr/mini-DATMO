"""只從 KITTI 的 data_odometry_velodyne.zip（84.8 GB）抓指定序列的點雲，不必下載整包。

原理：zip 的目錄（central directory）在檔案尾端，用 HTTP Range 先讀目錄（約 6 秒），
就知道每個 .bin 在大檔裡的位移。這個 zip 是 stored（不壓縮），所以每個 .bin 用一個 Range 請求直接取出。
多執行緒同時抓多個檔案，可以避開「單一連線被限速」。已存在且大小正確的檔案會跳過（可以中斷後重跑）。

用法：python tools/fetch_sequences.py 07 [05 06 ...] --out ../datasets/semantickitti --workers 16
"""
import argparse
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        d = get_range(self.url, self.pos, self.pos + n - 1)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def get_range(url, a, b, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise
            time.sleep(2 + 2 * t)


def fetch_member(info, out_root):
    path = os.path.join(out_root, info.filename)
    if os.path.exists(path) and os.path.getsize(path) == info.file_size:
        return 0
    # 本地檔頭 30 bytes + 檔名 + extra；extra 長度只有讀了檔頭才知道，多抓 1 KB 就夠
    a = info.header_offset
    blob = get_range(URL, a, a + 30 + len(info.filename.encode()) + 1024 + info.compress_size - 1)
    assert blob[:4] == b"PK\x03\x04", info.filename
    n_name, n_extra = struct.unpack("<HH", blob[26:30])
    start = 30 + n_name + n_extra
    data = blob[start:start + info.compress_size]
    assert len(data) == info.file_size, (info.filename, len(data), info.file_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "semantickitti"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
    want = [i for i in z.infolist() if i.compress_type == 0 and i.filename.endswith(".bin")
            and any(f"/sequences/{s}/velodyne/" in i.filename for s in args.seqs)]
    total = sum(i.file_size for i in want)
    print(f"{len(want)} files, {total/1e9:.2f} GB -> {os.path.abspath(args.out)}", flush=True)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_member, i, args.out) for i in want]
        for n, f in enumerate(as_completed(futs), 1):
            done += f.result()
            if n % 200 == 0 or n == len(want):
                dt = time.time() - t0
                print(f"  {n}/{len(want)} files  {done/1e9:.2f} GB new  {done/1e6/max(dt,1e-6):.1f} MB/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
