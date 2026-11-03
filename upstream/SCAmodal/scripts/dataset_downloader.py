import os
import requests
import re
from tqdm import tqdm

list_file = "dataset/COCOA/img_filenames_cocoa.txt"
output_dir = "dataset/COCOA/images"
os.makedirs(output_dir, exist_ok=True)

with open(list_file, "r") as f:
    filenames = [line.strip() for line in f.readlines() if line.strip()]

print(f"Bắt đầu xử lý và tải {len(filenames)} ảnh (Hỗ trợ Train/Val/Test)...")

for raw_filename in tqdm(filenames):
    save_path = os.path.join(output_dir, raw_filename)
    if os.path.exists(save_path): continue

    # 1. Xử lý sạch tên file
    clean_name = raw_filename.replace("./", "")
    
    # 2. Tìm tên ảnh gốc COCO (Hỗ trợ thêm 'test')
    # Regex: COCO_ + (train hoặc val hoặc test) + 2014 + số ID
    match = re.search(r'(COCO_(train|val|test)2014_\d+)', clean_name)
    
    if not match:
        print(f"\n⚠️ Không nhận diện được định dạng COCO: {clean_name}")
        continue
    
    coco_id = match.group(1) + ".jpg"
    subset = match.group(2) + "2014" # Ví dụ: test2014
    
    # 3. Tạo URL chính xác từ server MS COCO
    url = f"http://images.cocodataset.org/{subset}/{coco_id}"
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            # Một số ảnh test có thể nằm trong tập dev-test hoặc server khác, 
            # nhưng đa phần val/train/test2014 trên link này là chuẩn.
            print(f"\n❌ Thất bại: {url} (Status: {response.status_code})")
    except Exception as e:
        print(f"\n❌ Lỗi khi tải {raw_filename}: {e}")

print("\n✅ Hoàn thành tải toàn bộ ảnh!")
