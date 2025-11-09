# --------- هسته پیش‌پردازش فایل —--------
import os
import re
import glob
import logging

# کتابخانه‌های پردازش زبان طبیعی فارسی
from hazm import Normalizer, WordTokenizer

# کتابخانه‌های خواندن فایل
from docx import Document
import fitz  # PyMuPDF

# وارد کردن تنظیمات مسیرها
from preprocessing.config import MD_OUTPUT_DIR, TXT_OUTPUT_DIR, LOG_FILE

# وارد کردن توابع کمکی
from preprocessing.file_utils import sanitize_filename
from .ocr_utils import extract_pdf_text_and_images, extract_docx_text_and_images


class PersianTextPreprocessor:
    """کلاس مدیریت پیش‌پردازش متن فارسی"""

    def __init__(self):
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        # الگوی حذف نشانه‌گرهای آغاز خط (مانند •، –، * و غیره)
        self.pointer_pattern = re.compile(r'^[•–\-*►▪●○■□➤➔➣➢☑✔➽\s]*')
        # الگوی تمیز کردن فواصل متعدد
        self.space_pattern = re.compile(r'[ \t]+')

    def normalize_text(self, text: str) -> str:
        """نرمال‌سازی متن فارسی"""
        return self.normalizer.normalize(text)

    def tokenize_and_reconstruct(self, text: str) -> str:
        """توکن‌زنی و بازسازی مجدد جمله با فاصله مناسب"""
        tokens = self.tokenizer.tokenize(text)
        return " ".join(tokens)

    def clean_line(self, line: str) -> str:
        """پاک‌سازی یک خط متن: حذف فاصله‌های اضافی، نمادها و نرمال‌سازی"""
        line = re.sub(r"\s+", " ", line)  # تمیز کردن فواصل
        line = re.sub(r"[ـ_]", "", line)  # حذف کاراکترهای خاص
        line = self.pointer_pattern.sub('', line)  # حذف نشانه‌گرهای خط
        line = self.normalize_text(line)  # نرمال‌سازی
        line = self.tokenize_and_reconstruct(line)  # توکن‌زنی
        line = self.space_pattern.sub(' ', line).strip()  # تمیز کردن نهایی
        return line

    def group_lines_to_paragraphs(self, lines: list[str]) -> list[str]:
        """گروه‌بندی خطوط به صورت پاراگراف (بر اساس خطوط خالی)"""
        paragraphs, current = [], []
        for line in lines:
            stripped = line.strip()
            if not stripped and current:
                paragraphs.append(" ".join(current))
                current = []
            else:
                current.append(stripped)
        if current:
            paragraphs.append(" ".join(current))
        return paragraphs

    def preprocess_lines(self, lines: list[str]) -> list[str]:
        """پیش‌پردازش کامل لیست خطوط و تبدیل به پاراگراف‌های تمیز شده"""
        paragraphs = self.group_lines_to_paragraphs(lines)
        return [self.clean_line(p) for p in paragraphs if p.strip()]

def extract_text(filepath: str) -> list[str]:
    """استخراج متن خام از فایل‌های PDF، DOCX یا TXT"""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            with fitz.open(filepath) as doc:
                return [line for page in doc for line in page.get_text("text").splitlines()]
        elif ext == '.docx':
            doc = Document(filepath)
            return [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.readlines()
    except Exception as e:
        logging.error(f"Failed to extract from {filepath}: {e}")
    return []

def preprocess_file(filepath: str) -> tuple[str, str]:
    """پیش‌پردازش فایل و ذخیره آن به صورت .md و .txt"""
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == '.pdf':
            raw_lines = extract_pdf_text_and_images(filepath)
        elif ext == '.docx':
            raw_lines = extract_docx_text_and_images(filepath)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_lines = [line.strip() for line in f.readlines()]
        else:
            logging.warning(f"Unsupported file type: {filepath}")
            return "", ""
    except Exception as e:
        logging.error(f"Failed to extract from {filepath}: {e}")
        return "", ""

    if not raw_lines:
        logging.warning(f"Empty or unreadable file: {filepath}")
        return "", ""

    preprocessor = PersianTextPreprocessor()
    cleaned_lines = preprocessor.preprocess_lines(raw_lines)

    # تولید محتوای خروجی متنی و Markdown
    text_result = "\n".join(cleaned_lines)
    md_result = "\n\n".join(f"### {line}" if i == 0 else line for i, line in enumerate(cleaned_lines))

    # ساخت نام فایل خروجی با حذف تکرار غیرضروری نام پوشه
    folder_name = sanitize_filename(os.path.basename(os.path.dirname(filepath)))
    doc_name = os.path.splitext(os.path.basename(filepath))[0]

    # حذف تکرار پشت سر هم folder_name در ابتدای doc_name
    pattern = f"^{re.escape(folder_name)}_"
    while doc_name.startswith(folder_name + "_"):
        doc_name = re.sub(pattern, "", doc_name, count=1)

    base_name = sanitize_filename(f"{folder_name}_{doc_name}")

    # مسیرهای خروجی
    md_path = os.path.join(MD_OUTPUT_DIR, f"{base_name}.md")
    txt_path = os.path.join(TXT_OUTPUT_DIR, f"{base_name}.txt")

    # اطمینان از وجود پوشه‌های خروجی
    os.makedirs(MD_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

    # ذخیره فایل‌های خروجی
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_result)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text_result)

    logging.info(f"Processed {filepath} -> {md_path}, {txt_path}")
    return text_result, md_result



def get_processed_files() -> set:
    """دریافت لیست فایل‌هایی که قبلاً پردازش شده‌اند"""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

def log_processed_file(filepath: str):
    """ثبت فایل پردازش شده در فایل لاگ"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(filepath + "\n")