"""TV1 – Hàm dùng chung cho các script thí nghiệm."""
from pathlib import Path

import cv2


def imwrite_unicode(path, img):
    """cv2.imwrite KHÔNG ghi được đường dẫn có dấu tiếng Việt trên Windows (trả False, không báo lỗi).
    Mã hóa ảnh trong bộ nhớ rồi để NumPy ghi file."""
    ext = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Không mã hóa được ảnh: {path}")
    buf.tofile(str(path))


def label(img, text):
    """Thêm dải tiêu đề đen phía trên ảnh (ảnh xám được đổi sang BGR)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out
