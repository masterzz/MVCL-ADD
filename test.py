#两行代码搞定GitHub上传！
# git config --global http.proxy http://127.0.0.1:7890
# git config --global https.proxy http://127.0.0.1:7890
import torch
import time
# 假设你的类定义在 multiView_model.py 中，如果文件名不同请修改这里
from MVCL.multiView_model import DynamicTrajectoryBranch

def verify_module():
    print("=" * 40)
    print("🚀 开始验证 DynamicTrajectoryBranch 模块")
    print("=" * 40)

    # 1. 定义模拟参数 (模拟 WavLM Base 的输出)
    BATCH_SIZE = 4
    SEQ_LEN = 149   # 假设音频长度对应的帧数
    INPUT_DIM = 768 # 输入特征维度
    HIDDEN_DIM = 256 # 内部隐藏层维度

    # 2. 实例化模型
    
    print(f"🔨 正在初始化模型 (Input={INPUT_DIM}, Hidden={HIDDEN_DIM})...")
    model = DynamicTrajectoryBranch(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=2
    )

    # 3. 构造伪造输入数据 (Random Tensor)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    print(f"📥 输入数据形状: {dummy_input.shape} (B, T, D)")

    # ---------------------------------------------------------
    # 测试 A: CPU 维度验证
    # ---------------------------------------------------------
    print("\n[测试 A] CPU 前向传播 & 维度检查...")
    try:
        # 前向传播
        final_feat, lstm_out = model(dummy_input)
        
        # 打印输出形状
        print(f"   --> final_feat (全局特征): {final_feat.shape}")
        print(f"   --> lstm_out   (序列特征): {lstm_out.shape}")

        # 自动断言检查 (Assertion)
        # 预期 final_feat: (B, HIDDEN_DIM) -> (4, 256)
        assert final_feat.shape == (BATCH_SIZE, HIDDEN_DIM), \
            f"❌ 全局特征维度错误! 预期 {(BATCH_SIZE, HIDDEN_DIM)}, 实际 {final_feat.shape}"
        
        # 预期 lstm_out: (B, T, HIDDEN_DIM * 2) -> (4, 149, 512) (因为是双向 Bi-GRU)
        assert lstm_out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM * 2), \
            f"❌ 序列特征维度错误! 预期 {(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM * 2)}, 实际 {lstm_out.shape}"

        print("✅ CPU 维度验证通过！逻辑无误。")

    except Exception as e:
        print(f"❌ CPU 测试失败: {e}")
        return # CPU 挂了就不测 GPU 了

    # ---------------------------------------------------------
    # 测试 B: GPU (RTX 5070 Ti) 兼容性验证
    # ---------------------------------------------------------
    print("\n[测试 B] GPU (CUDA) 兼容性测试...")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            
            # 将模型和数据搬运到 GPU
            model = model.to(device)
            dummy_input_gpu = dummy_input.to(device)
            
            # 记录时间跑一下
            start_t = time.time()
            _ = model(dummy_input_gpu)
            end_t = time.time()
            
            print(f"   设备名称: {torch.cuda.get_device_name(0)}")
            print(f"   推理耗时: {(end_t - start_t) * 1000:.2f} ms")
            print("✅ GPU 运行成功！Blackwell 架构兼容性正常。")
            
        except RuntimeError as e:
            print(f"❌ GPU 运行失败 (可能是显存或版本问题): \n{e}")
    else:
        print("⚠️ 未检测到 GPU，跳过此测试。")

if __name__ == "__main__":
    verify_module()

