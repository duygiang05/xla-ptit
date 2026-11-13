"""TV1 – Chia bộ FIRESENSE vào development/integration/test và ghi data/manifest.csv.

Video KHÔNG nằm trên git. Để mọi thành viên có cùng bộ dữ liệu:
  1. Tải 2 file từ https://zenodo.org/records/836749 (CC BY 4.0):
       fire_videos.1406.zip, smoke_videos.1407.zip
  2. Giải nén vào  data/raw/firesense/fire/{pos,neg}/  và  data/raw/firesense/smoke/{pos,neg}/
  3. Chạy từ thư mục gốc repo:
       python experiments/tv1/prepare_firesense.py            # sao chép (giữ data/raw)
       python experiments/tv1/prepare_firesense.py --move     # di chuyển (tiết kiệm dung lượng)
     Thêm --manifest-only để chỉ ghi lại data/manifest.csv.

Tiêu chí chọn (TV1, 2026-11-12) – chỉ giữ video dùng được cho CẢ pipeline:
  - camera cố định: độ trượt khung (phaseCorrelate, experiments/tv1/inspect_videos.py) p90 <= 3 px
  - không cắt cảnh, không có lớp vẽ đè kết quả của thuật toán khác
  Bị loại (11 lửa + 8 khói): fire neg 1,4,5,6,7,8,9,12, fire pos 2,8,9; smoke pos 01,04,05,09–13.
Chia theo NGUỒN: video cùng cảnh/cùng nguồn (cột original_video_id) nằm chung một tập.
Nhãn has_fire/has_smoke đặt theo ảnh mẫu – cần người xem lại video để xác nhận.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "firesense"
SOURCE_URL = "https://zenodo.org/records/836749"
LICENSE = "CC BY 4.0 (FIRESENSE, doi:10.5281/zenodo.836749)"
OWNER = "TV1"

# video_id, file gốc (trong data/raw/firesense), split, has_fire, has_smoke, original_video_id, notes
VIDEOS = [
    # ---------------- development (chỉnh ngưỡng)
    ("fs_smoke_pos02", "smoke/pos/testpos02.818.avi", "development", 1, 1, "", "NIST building fire, fire in window + smoke; NIST logo"),
    ("fs_fire_pos01", "fire/pos/posVideo1.868.avi", "development", 1, 1, "", "burning wagon, black smoke; small FPS text top-left"),
    ("fs_fire_pos03", "fire/pos/posVideo3.872.avi", "development", 1, 0, "", "brush fire close-up; slight shake 1.2 px"),
    ("fs_fire_pos04", "fire/pos/posVideo4.873.avi", "development", 1, 0, "fs_fireplace", "fireplace, dark brick background"),
    ("fs_fire_pos05", "fire/pos/posVideo5.874.avi", "development", 1, 0, "fs_fireplace", "fireplace"),
    ("fs_smoke_pos06", "smoke/pos/testpos06.822.avi", "development", 0, 1, "fs_bucket_yard", "smoke from bucket, people walking; fisheye; FPS text"),
    ("fs_smoke_pos07", "smoke/pos/testpos07.823.avi", "development", 0, 1, "fs_bucket_yard", "smoke from bucket, person; fisheye; FPS text"),
    ("fs_fire_neg02", "fire/neg/negsVideo2.859.avi", "development", 0, 0, "fs_night_street", "night street, car headlights"),
    ("fs_fire_neg03", "fire/neg/negsVideo3.860.avi", "development", 0, 0, "fs_night_street", "night street, headlights, people"),
    ("fs_fire_neg11", "fire/neg/negsVideo11.1073.avi", "development", 0, 0, "", "red car passing (fire color + motion)"),
    ("fs_smoke_neg01", "smoke/neg/testneg01.807.avi", "development", 0, 0, "", "tree with pink blossoms, clouds"),
    ("fs_smoke_neg06", "smoke/neg/testneg06.812.avi", "development", 0, 0, "fs_highway", "highway traffic"),
    ("fs_smoke_neg08", "smoke/neg/testneg08.814.avi", "development", 0, 0, "", "red/yellow flags waving (fire color + motion)"),
    ("fs_smoke_neg09", "smoke/neg/testneg09.815.avi", "development", 0, 0, "fs_highway", "highway traffic, orange truck"),
    # ---------------- integration (ghép thử pipeline)
    ("fs_fire_pos06", "fire/pos/posVideo6.875.avi", "integration", 1, 1, "", "small field fire with smoke; FPS text"),
    ("fs_fire_pos07", "fire/pos/posVideo7.876.avi", "integration", 1, 0, "", "small fire near wall, field"),
    ("fs_smoke_pos03", "smoke/pos/testpos03.819.avi", "integration", 0, 1, "", "smoke over field; overexposed, low quality"),
    ("fs_fire_neg10", "fire/neg/negsVideo10.1072.avi", "integration", 0, 0, "", "office, person with orange bag (fire color + motion)"),
    ("fs_smoke_neg02", "smoke/neg/testneg02.808.avi", "integration", 0, 0, "", "people on grey asphalt (smoke color + motion)"),
    ("fs_smoke_neg05", "smoke/neg/testneg05.811.avi", "integration", 0, 0, "", "reeds moving in wind"),
    # ---------------- test (chỉ chạy MỘT lần sau khi khóa cấu hình)
    ("fs_fire_pos10", "fire/pos/posVideo10.869.avi", "test", 1, 1, "fs_ultimatechase", "forest fire, heavy smoke; watermark UltimateChase.com"),
    ("fs_fire_pos11", "fire/pos/posVideo11.870.avi", "test", 1, 0, "fs_ultimatechase", "forest ground fire, firefighters; watermark"),
    ("fs_smoke_pos08", "smoke/pos/testpos08.824.avi", "test", 1, 1, "", "smoke with red flare at base"),
    ("fs_fire_neg13", "fire/neg/negsVideo13.1075.avi", "test", 0, 0, "fs_station", "station crowd; camera OSD timestamp"),
    ("fs_fire_neg14", "fire/neg/negsVideo14.1076.avi", "test", 0, 0, "fs_station", "train platform; only 1.9 s"),
    ("fs_fire_neg15", "fire/neg/negsVideo15.1078.avi", "test", 0, 0, "", "bridge, orange truck; only 2.9 s; 1600x1200"),
    ("fs_fire_neg16", "fire/neg/negsVideo16.1077.avi", "test", 0, 0, "", "lake, person in red shirt; 1600x1200"),
    ("fs_smoke_neg03", "smoke/neg/testneg03.809.avi", "test", 0, 0, "", "office; only first 182/748 frames readable (~6 s)"),
    ("fs_smoke_neg04", "smoke/neg/testneg04.810.avi", "test", 0, 0, "", "people near fence, trees; 113 s"),
    ("fs_smoke_neg07", "smoke/neg/testneg07.813.avi", "test", 0, 0, "", "trees, clouds, blue sky"),
]
FIELDS = ["video_id", "filename", "split", "has_fire", "has_smoke", "source_url", "license",
          "original_video_id", "owner", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true", help="di chuyển thay vì sao chép")
    ap.add_argument("--manifest-only", action="store_true", help="chỉ ghi data/manifest.csv")
    args = ap.parse_args()

    rows, missing = [], []
    for vid, src, split, fire, smoke, orig, notes in VIDEOS:
        filename = f"{vid}{Path(src).suffix}"
        dst = ROOT / "data" / split / filename
        if not args.manifest_only and not dst.exists():
            s = RAW / src
            if not s.exists():
                missing.append(src)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                (shutil.move if args.move else shutil.copy2)(str(s), str(dst))
        rows.append({"video_id": vid, "filename": filename, "split": split, "has_fire": fire,
                     "has_smoke": smoke, "source_url": SOURCE_URL, "license": LICENSE,
                     "original_video_id": orig or vid, "owner": OWNER, "notes": notes})

    # Kiểm tra quy tắc chia theo nguồn: một original_video_id chỉ nằm trong MỘT split
    splits_of = {}
    for r in rows:
        splits_of.setdefault(r["original_video_id"], set()).add(r["split"])
    leaks = {k: v for k, v in splits_of.items() if len(v) > 1}
    if leaks:
        sys.exit(f"LỖI: cùng nguồn nhưng khác split: {leaks}")

    with (ROOT / "data" / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for split in ("development", "integration", "test"):
        sel = [r for r in rows if r["split"] == split]
        count = lambda f, s: sum(r["has_fire"] == f and r["has_smoke"] == s for r in sel)  # noqa: E731
        print(f"{split:12s} {len(sel):2d} video | cả hai {count(1, 1)} | chỉ lửa {count(1, 0)} | "
              f"chỉ khói {count(0, 1)} | âm tính {count(0, 0)}")
    if missing:
        print("THIẾU file gốc (chưa tải/giải nén?):", *missing, sep="\n  ")
    print("Đã ghi data/manifest.csv")


if __name__ == "__main__":
    main()
