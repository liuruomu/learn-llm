import os
from huggingface_hub import snapshot_download

# 1. 临时设置镜像站环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 配置下载参数
repo_id = "sentence-transformers/all-MiniLM-L6-v2"
local_dir = "./all-MiniLM-L6-v2"

print(f"🚀 开始从镜像站下载模型: {repo_id} ...")

try:
    # 3. 执行下载
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 必须为 False，否则拷到内网全是快捷方式
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略掉不需要的权重格式，节省空间
    )
    print(f"✅ 下载完成！文件夹位于: {os.path.abspath(local_dir)}")
except Exception as e:
    print(f"❌ 下载失败: {e}")