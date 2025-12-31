import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from MVCL import MultiViewModel_lit, MultiViewModel

# 引入回调函数 (请确保 callbacks.py 在同级目录下)
try:
    from callbacks import EER_Callback, BinaryAUC_Callback, BinaryACC_Callback
except ImportError:
    print("⚠️ 警告: 找不到 callbacks.py，将跳过 EER/ACC 计算回调")
    EER_Callback = None
    BinaryACC_Callback = None

# ================= 1. 全局配置区域 =================
# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 路径配置
DATA_ROOT = r"D:\learn\MVCL-ADD\download\archive\LA"
LIST_SAVE_DIR = "data_lists"

# 训练超参数
BATCH_SIZE = 16         # 显存允许可调大
MAX_EPOCHS = 5          # 训练轮数
NUM_WORKERS = 0         # Windows 必须为 0

# 模型配置参数
MVCL_CFG = Namespace(
    use_inner_CL=1,
    use_inter_CL=1,
    use_cls_loss_1_2=1,
    use_fusion=1,
    use_fusion1D=1,
    use_fusion2D=1,
    use_mse_loss=0,
    only_1D=0,
    only_2D=0,
    drop_layer=0.0,
    w_con=1.0,
    w_cls=1.0,
)

# ================= 2. 工具函数与类定义 =================

def check_environment():
    """检查 GPU 和必要库"""
    print("-" * 30)
    if torch.cuda.is_available():
        print(f"✅ GPU 就绪: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 未检测到 GPU，将使用 CPU 运行 (极慢)")
    
    try:
        import soundfile
        print("✅ soundfile 库已检测到")
    except ImportError:
        raise ImportError("❌ 严重错误: 请先运行 `pip install soundfile`")
    print("-" * 30)

def generate_list(protocol_name, audio_dir_name, output_name):
    """通用的列表生成函数 (支持 Train, Dev, Eval)"""
    if not os.path.exists(LIST_SAVE_DIR):
        os.makedirs(LIST_SAVE_DIR)

    protocol_path = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", protocol_name)
    audio_dir = os.path.join(DATA_ROOT, audio_dir_name, "flac")
    output_path = os.path.join(LIST_SAVE_DIR, output_name)
    
    # 检查是否已存在，避免重复生成
    if os.path.exists(output_path):
        print(f"ℹ️ 列表已存在，跳过生成: {output_path}")
        return output_path

    if not os.path.exists(protocol_path):
        print(f"❌ 错误：找不到协议文件: {protocol_path}")
        return None

    print(f"🚀 正在生成列表: {output_name} ...")
    count = 0
    with open(protocol_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(' ')
            # 协议格式: SPEAKER_ID AUDIO_NAME ... KEY
            # 例如: LA_0079 LA_T_1138215 - - bonafide
            audio_name = parts[1]
            label_str = parts[4] # 第5列是标签
            
            # 转换标签: bonafide(真)=1, spoof(假)=0
            label = 1 if label_str == 'bonafide' else 0
            
            full_path = os.path.join(audio_dir, audio_name + '.flac')
            f_out.write(f"{full_path} {label}\n")
            count += 1
            
    print(f"✅ 生成完毕！共 {count} 条数据。")
    return output_path

class ASVspoofDataset(Dataset):
    """
    自定义 Dataset 类
    使用 soundfile 直接读取音频，绕过 torchaudio 在 Windows 下的后端问题
    """
    def __init__(self, list_path, max_len=48000):
        self.data_list = []
        self.max_len = max_len
        
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"找不到列表文件: {list_path}")
            
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    self.data_list.append((parts[0], int(parts[1])))
        
        print(f"📦 Dataset 加载完成 [{os.path.basename(list_path)}]: {len(self.data_list)} 样本")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, label = self.data_list[idx]
        try:
            # 1. 使用 soundfile 读取 (绕过 torchaudio 后端问题)
            speech, sample_rate = sf.read(audio_path)
            
            # 2. 转 Tensor
            waveform = torch.from_numpy(speech).float()
            
            # 3. 维度调整: 确保形状为 [1, T]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0) # [T] -> [1, T]
            elif waveform.dim() == 2:
                waveform = waveform.t()          # [T, C] -> [C, T]

            # 4. 长度统一处理 (Pad or Crop)
            curr_len = waveform.shape[1]
            if curr_len < self.max_len:
                # 补零
                pad_width = self.max_len - curr_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_width))
            else:
                # 截断
                waveform = waveform[:, :self.max_len]
                
            return {
                "audio": waveform, 
                "label": label, 
                "sample_rate": sample_rate
            }
            
        except Exception as e:
            print(f"⚠️ 读取失败 [{audio_path}]: {e}")
            # 返回全0数据防止崩溃
            return {
                "audio": torch.zeros(1, self.max_len), 
                "label": label, 
                "sample_rate": 16000
            }

# ================= 3. 主执行逻辑 =================

if __name__ == "__main__":
    print("=" * 40)
    print("🚀 开始执行 MVCL 全流程脚本 (训练 + 测试)")
    print("=" * 40)

    # --- 步骤 1: 环境检查 ---
    check_environment()

    # --- 步骤 2: 生成所有数据集列表 ---
    print("\n[Step 1/5] 准备数据列表...")
    
    # 2.1 训练集
    train_list = generate_list(
        protocol_name="ASVspoof2019.LA.cm.train.trn.txt", 
        audio_dir_name="ASVspoof2019_LA_train", 
        output_name="train_list.txt"
    )
    # 2.2 验证集
    dev_list = generate_list(
        protocol_name="ASVspoof2019.LA.cm.dev.trl.txt", 
        audio_dir_name="ASVspoof2019_LA_dev", 
        output_name="dev_list.txt"
    )
    # 2.3 测试集
    eval_list = generate_list(
        protocol_name="ASVspoof2019.LA.cm.eval.trl.txt", 
        audio_dir_name="ASVspoof2019_LA_eval", 
        output_name="eval_list.txt"
    )

    if not (train_list and dev_list and eval_list):
        exit("❌ 数据列表生成失败，请检查路径配置！")

    # --- 步骤 3: 准备 DataLoaders ---
    print("\n[Step 2/5] 实例化 DataLoaders...")
    
    # 训练集 Loader
    train_loader = DataLoader(
        ASVspoofDataset(train_list, max_len=48000),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    # 验证集 Loader
    val_loader = DataLoader(
        ASVspoofDataset(dev_list, max_len=48000),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    # 测试集 Loader
    test_loader = DataLoader(
        ASVspoofDataset(eval_list, max_len=48000),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --- 步骤 4: 初始化模型 ---
    print("\n[Step 3/5] 初始化模型...")
    mvcl_lit = MultiViewModel_lit(cfg=MVCL_CFG)

    # --- 步骤 5: 配置 Callbacks 和 Trainer ---
    print("\n[Step 4/5] 配置 Trainer...")
    
    # 定义回调列表
    callbacks = [
        # 模型检查点：监控 val-eer，保存效果最好的模型
        ModelCheckpoint(
            monitor="val-eer", 
            mode="min", 
            save_top_k=1, 
            filename="{epoch}-{val-eer:.4f}"
        )
    ]
    # 添加 EER 和 ACC 计算回调 (如果导入成功)
    if EER_Callback:
        callbacks.append(EER_Callback(batch_key="label", output_key="logit"))
    if BinaryACC_Callback:
        callbacks.append(BinaryACC_Callback(batch_key="label", output_key="logit"))

    # 初始化 Trainer
    trainer = Trainer(
        accelerator="gpu", 
        devices=[0], 
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=CSVLogger(save_dir="./logs", name="MVCL_Experiment", version=None),
        log_every_n_steps=50,
        enable_checkpointing=True
    )

    # --- 步骤 6: 开始训练与测试 ---
    print("\n" + "=" * 40)
    print("🔥 [Step 5/5] 引擎启动！开始训练 (Training)...")
    print("=" * 40)
    
    # 1. 训练 (会自动使用 train_loader 和 val_loader)
    trainer.fit(mvcl_lit, train_loader, val_loader)

    print("\n" + "=" * 40)
    print("🚀 训练结束！开始测试 (Testing)...")
    print("=" * 40)
    
    # 2. 测试 (使用刚才训练好的模型测试 eval 集)
    # 'best' 会自动加载 checkpoint 中效果最好的模型权重
    trainer.test(mvcl_lit, test_loader, ckpt_path='best')
    
    print("\n🎉 所有流程圆满结束！请查看 ./logs 文件夹获取日志。")