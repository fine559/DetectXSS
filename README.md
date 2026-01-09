# XSS攻击检测系统

基于多模型深度学习的XSS攻击检测系统，使用XGBoost、BiLSTM、Transformer三种模型进行集成检测。

## 功能特性

- **多模型检测**: 集成XGBoost、BiLSTM、Transformer三种模型
- **加权平均**: 使用加权平均进行模型融合
- **Web界面**: 提供友好的Web检测界面
- **历史记录**: 保存检测历史和统计数据
- **MySQL存储**: 使用MySQL存储检测记录

## 项目结构

```
DetectXSS/
├── app.py                 # Flask主应用
├── database.py            # 数据库管理
├── data_processor.py      # 数据预处理
├── models.py              # 模型定义（XGBoost/BiLSTM/Transformer）
├── ensemble.py            # 模型集成
├── train_models.py        # 模型训练脚本
├── requirements.txt       # 依赖包
├── templates/             # HTML模板
│   └── index.html        # 主页面
└── models/               # 模型保存目录（训练后生成）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据库配置

默认MySQL配置:
- 主机: localhost
- 用户: root
- 密码: 123456
- 数据库: xss_detection

如需修改，请编辑 `database.py` 中的配置。

## 快速开始

### 1. 训练模型

首先训练模型（会自动生成示例数据）:

```bash
python train_models.py
```

测试已训练的模型:

```bash
python train_models.py --test
```

### 2. 启动Web服务

```bash
python app.py
```

服务将在 http://localhost:5000 启动

### 3. 访问检测页面

在浏览器中访问: http://localhost:5000

## API接口

### 检测XSS攻击

```bash
POST /api/detect
Content-Type: application/json

{
  "text": "<script>alert('xss')</script>"
}
```

响应:

```json
{
  "is_xss": true,
  "xgboost_prob": 0.95,
  "bilstm_prob": 0.92,
  "transformer_prob": 0.97,
  "ensemble_prob": 0.95,
  "message": "XSS攻击"
}
```

### 获取统计数据

```bash
GET /api/statistics
```

响应:

```json
{
  "total": 100,
  "xss_count": 45,
  "normal_count": 55
}
```

### 获取检测历史

```bash
GET /api/history?limit=100
```

### 健康检查

```bash
GET /api/health
```

## 模型说明

### XGBoost
- 基于梯度提升决策树
- 使用TF-IDF特征和手工特征
- 适合处理结构化特征

### BiLSTM
- 双向长短期记忆网络
- 使用字符级分词
- 能够捕捉序列信息

### Transformer
- 自注意力机制
- 位置编码
- 捕捉长距离依赖

### 集成策略
- 加权平均: XGBoost(30%) + BiLSTM(35%) + Transformer(35%)
- 支持投票法作为备选策略

## 注意事项

1. 首次运行前请确保MySQL服务已启动
2. 训练模型需要较长时间，建议在训练前确保系统资源充足
3. 模型文件会保存在 `models/` 目录下
4. 如果使用自定义数据集，请确保数据格式正确

## 许可证

MIT License
