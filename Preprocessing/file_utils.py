# --------- کتابخانه‌های مورد نیاز —--------
import os
import re
import zipfile
import shutil
import glob
import logging

try:
    import rarfile
except ImportError:
    rarfile = None

from preprocessing.config import UPLOAD_DIR

# --------- راه‌اندازی پوشه‌ها —--------
def setup_directories():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------- تمیز کردن نام فایل از کاراکترهای نامعتبر —--------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?~:"<>|$~]', '_', name)

# --------- تابع برای یافتن شماره بعدی برای فایل‌های single_###_ —--------
def get_next_single_counter(target_dir):
    pattern = os.path.join(target_dir, "single_???_*")
    files = glob.glob(pattern)
    numbers = []
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"single_(\d{3})_", basename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 1

# --------- استخراج ZIP —--------
def extract_zip(file_path: str, target_dir: str):
    try:
        base_folder = sanitize_filename(os.path.splitext(os.path.basename(file_path))[0])
        extract_dir = os.path.join(target_dir, base_folder)
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(file_path)
        logging.info(f"✅ Extracted ZIP: {file_path}")

        rename_inside_folder(extract_dir, base_folder)

    except zipfile.BadZipFile:
        logging.error(f"❌ Bad ZIP file: {file_path}")

# --------- استخراج RAR —--------
def extract_rar(file_path: str, target_dir: str):
    if not rarfile:
        logging.warning("❗ RAR file support requires `rarfile` package.")
        return
    try:
        base_folder = sanitize_filename(os.path.splitext(os.path.basename(file_path))[0])
        extract_dir = os.path.join(target_dir, base_folder)
        os.makedirs(extract_dir, exist_ok=True)

        with rarfile.RarFile(file_path, 'r') as rar:
            rar.extractall(extract_dir)
        os.remove(file_path)
        logging.info(f"✅ Extracted RAR: {file_path}")

        rename_inside_folder(extract_dir, base_folder)

    except rarfile.Error as e:
        logging.error(f"❌ Failed to extract RAR: {file_path} | Error: {e}")

# --------- انتقال پوشه —--------
def move_folder(folder_path: str, target_dir: str):
    try:
        base_name = sanitize_filename(os.path.basename(folder_path))
        target_path = os.path.join(target_dir, base_name)
        os.makedirs(target_path, exist_ok=True)

        for dirpath, _, filenames in os.walk(folder_path):
            for file in filenames:
                src_file = os.path.join(dirpath, file)
                new_name = f"{base_name}_{sanitize_filename(file)}"
                dst_file = os.path.join(target_path, new_name)

                temp_path = dst_file + ".tmp"
                shutil.copyfile(src_file, temp_path)
                shutil.move(temp_path, dst_file)

        logging.info(f"✅ Moved and renamed contents of folder: {folder_path}")
    except Exception as e:
        logging.error(f"❌ Error moving folder: {folder_path} | {e}")

# --------- انتقال فایل تکی —--------
def move_supported_file(file_path: str, target_dir: str, counter: int) -> int:
    ext = os.path.splitext(file_path)[1].lower()
    supported = ['.pdf', '.docx', '.txt']
    if ext in supported:
        sanitized_name = sanitize_filename(os.path.basename(file_path))
        numbered_name = f"single_{counter:03d}_{sanitized_name}"
        target_path = os.path.join(target_dir, numbered_name)

        temp_path = target_path + ".tmp"
        shutil.copyfile(file_path, temp_path)
        shutil.move(temp_path, target_path)

        logging.info(f"✅ Moved file: {target_path}")
        return counter + 1
    else:
        logging.warning(f"⚠️ Unsupported file skipped: {file_path}")
        return counter

# --------- تغییر نام فایل‌های داخل یک پوشه —--------
def rename_inside_folder(folder_path: str, folder_base: str):
    for file in os.listdir(folder_path):
        old_path = os.path.join(folder_path, file)
        if os.path.isfile(old_path):
            new_name = f"{folder_base}_{sanitize_filename(file)}"
            new_path = os.path.join(folder_path, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                logging.info(f"🔁 Renamed inside folder: {old_path} ➜ {new_path}")

# --------- تابع اصلی مدیریت فایل‌های آپلود شده —--------
def handle_uploaded_items(paths: list[str]):
    setup_directories()
    counter = get_next_single_counter(UPLOAD_DIR)

    for path in paths:
        if os.path.isdir(path):
            move_folder(path, UPLOAD_DIR)
        elif path.endswith('.zip'):
            extract_zip(path, UPLOAD_DIR)
        elif path.endswith('.rar'):
            extract_rar(path, UPLOAD_DIR)
        else:
            counter = move_supported_file(path, UPLOAD_DIR, counter)
