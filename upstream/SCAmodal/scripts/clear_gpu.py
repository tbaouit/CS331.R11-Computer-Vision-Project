import torch
import gc
import sys

def super_clear_gpu():
    print("--- Đang quét và tiêu diệt các tham chiếu Model/Tensor ---")
    
    # 1. Quét toàn bộ biến toàn cục (Globals)
    for name in list(globals().keys()):
        if name.startswith('_'): continue # Bỏ qua biến nội bộ
        obj = globals()[name]
        
        # Nếu là Model, Pipeline hoặc Tensor lớn
        if isinstance(obj, (torch.nn.Module, torch.Tensor)) or 'Pipeline' in str(type(obj)):
            print(f"Bắt giữ và xóa: {name}")
            del globals()[name]
            
    # 2. Xóa bộ nhớ đệm của hệ thống
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # 3. Thông báo trạng thái
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"--- THÀNH CÔNG ---")
        print(f"GPU trống: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")

super_clear_gpu()
