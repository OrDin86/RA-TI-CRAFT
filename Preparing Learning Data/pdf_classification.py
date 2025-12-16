import os
import shutil
import fitz  # PyMuPDF
from pathlib import Path

# ===== 원본 PDF 경로 =====
SRC_DIR = r"E:\pdf"

# ===== 복사 후 분류할 경로 =====
BASE_DIR = r"E:\pdf\org_pdf"
TEXT_DIR = os.path.join(BASE_DIR, "텍스트PDF")
IMAGE_DIR = os.path.join(BASE_DIR, "이미지PDF")
MIXED_DIR = os.path.join(BASE_DIR, "혼합PDF")
LOG_PATH  = os.path.join(BASE_DIR, "pdf_classify_log.txt")

# 분류 폴더 생성
for folder in [TEXT_DIR, IMAGE_DIR, MIXED_DIR]:
    os.makedirs(folder, exist_ok=True)

def unique_copy(src_path: str, dst_dir: str):
    """
    dst_dir 안에 같은 파일명이 있으면 _1, _2 같이 번호를 붙여서 복사.
    메타데이터 보존(copy2).
    """
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src_path)
    stem, ext = os.path.splitext(name)
    dst_path = os.path.join(dst_dir, name)
    idx = 1
    while os.path.exists(dst_path):
        dst_path = os.path.join(dst_dir, f"{stem}_{idx}{ext}")
        idx += 1
    shutil.copy2(src_path, dst_path)
    return dst_path

def classify_pdf(pdf_path: str):
    """
    PDF 유형 판별:
    - 텍스트 기반: 텍스트(단어) 존재 & 이미지 없음
    - 이미지 스캔: 텍스트 없음 & 이미지 존재
    - 혼합형: 둘 다 있거나 일부 페이지만 텍스트/이미지
    """
    has_text = False
    has_image = False

    with fitz.open(pdf_path) as doc:
        for page in doc:
            words = page.get_text("words")
            if words:
                has_text = True
            if page.get_images(full=True):
                has_image = True
            if has_text and has_image:
                break

    if has_text and not has_image:
        return "text"
    elif not has_text and has_image:
        return "image"
    else:
        return "mixed"

def main():
    total, ok, err = 0, 0, 0
    with open(LOG_PATH, "w", encoding="utf-8") as logf:
        for root, _, files in os.walk(SRC_DIR):
            for fname in files:
                if not fname.lower().endswith(".pdf"):
                    continue
                total += 1
                src = os.path.join(root, fname)
                try:
                    cls = classify_pdf(src)
                    if cls == "text":
                        dst = unique_copy(src, TEXT_DIR)
                        logf.write(f"[TEXT ] {src} -> {dst}\n")
                    elif cls == "image":
                        dst = unique_copy(src, IMAGE_DIR)
                        logf.write(f"[IMAGE] {src} -> {dst}\n")
                    else:
                        dst = unique_copy(src, MIXED_DIR)
                        logf.write(f"[MIXED] {src} -> {dst}\n")
                    ok += 1
                except Exception as e:
                    err += 1
                    logf.write(f"[ERROR] {src} :: {repr(e)}\n")

    print(f"완료: 총 {total}개 / 성공 {ok}개 / 오류 {err}개")
    print(f"로그 파일 위치: {LOG_PATH}")

if __name__ == "__main__":
    main()