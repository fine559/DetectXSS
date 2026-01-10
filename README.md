# XSS攻击检测系统

基于多模型深度学习的XSS攻击检测系统，使用XGBoost、BiLSTM、Transformer三种模型进行集成检测。

## 功能特性

- **多模型检测**: 集成XGBoost、BiLSTM、Transformer三种模型
- **加权平均集成**: 使用加权平均进行模型融合，提高检测准确率
- **Web界面**: 提供友好的Web检测界面
- **单次/批量检测**: 支持单次检测和批量检测
- **检测结果详情**: 提供详细的分析报告，包括风险等级、攻击类型、危险特征高亮
- **检测历史**: 保存检测历史，支持分页浏览、单条删除和批量删除
- **监控仪表盘**: 实时展示检测统计、趋势图和模型性能
- **模型性能分析**: 对比各模型性能指标，展示训练历史曲线
- **模型训练与评估**: 支持在线训练模型并评估性能
- **MySQL存储**: 使用MySQL存储检测记录、训练数据和模型信息
- **自动数据清洗**: 训练历史数据仅在完整训练完成后写入，避免重复数据

## 项目结构

```
DetectXSS/
├── app.py                 # Flask主应用
├── config.py             # 配置文件
├── database.py           # 数据库管理
├── data_processor.py     # 数据预处理
├── models.py             # 模型定义（XGBoost/BiLSTM/Transformer）
├── ensemble.py           # 模型集成
├── train_models.py       # 模型训练脚本
├── requirements.txt      # 依赖包
├── templates/            # HTML模板
│   ├── index.html        # 主页面（单次/批量检测、监控仪表盘、检测历史）
│   ├── detail.html       # 检测结果详情分析页面
│   ├── analysis.html     # 模型性能对比分析页面
│   └── training.html     # 模型训练与评估页面
└── models/              # 模型保存目录（训练后生成）
    ├── xgboost_model.pkl
    ├── bilstm_model.h5
    ├── transformer_model.h5
    └── metrics.json      # 模型性能指标
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

### 3. 访问Web界面

在浏览器中访问: http://localhost:5000

系统提供以下功能页面：

- **单次检测**: 输入单个文本进行XSS检测
- **批量检测**: 一次性检测多个文本（最多100条）
- **检测规则**: 查看各模型权重和集成方法
- **监控仪表盘**: 查看检测统计、趋势图和模型性能
- **检测历史**: 浏览历史检测记录，支持删除操作
- **模型性能分析**: 对比各模型性能指标和训练曲线
- **检测结果详情**: 获取详细的风险分析和特征匹配
- **模型训练与评估**: 在线训练模型并评估性能

## 快速访问链接

从主页可直接访问以下功能：

- `/analysis` - 模型性能分析
- `/detail` - 检测结果详情
- `/training` - 模型训练与评估

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

### 详细分析

```bash
POST /api/detect-detail
Content-Type: application/json

{
  "text": "<script>alert('xss')</script>"
}
```

响应包含风险等级、攻击类型、特征匹配等信息。

### 批量检测

```bash
POST /api/batch-detect
Content-Type: application/json

{
  "texts": ["<script>alert('xss')</script>", "Hello world"]
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

### 获取检测历史（分页）

```bash
GET /api/history?page=1&page_size=10
```

### 删除检测记录

```bash
DELETE /api/detection-records/<record_id>
```

### 批量删除检测记录

```bash
POST /api/detection-records/batch-delete
Content-Type: application/json

{
  "record_ids": [1, 2, 3]
}
```

### 获取仪表盘数据

```bash
GET /api/dashboard?days=7
```

### 获取模型性能数据

```bash
GET /api/model-metrics
```

### 获取训练历史

```bash
GET /api/training-history?model_name=xgboost
```

### 获取检测规则配置

```bash
GET /api/rules
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

1. **数据库配置**: 首次运行前请确保MySQL服务已启动，并根据实际情况修改 `database.py` 中的数据库配置
2. **模型训练**: 训练模型需要较长时间，建议在训练前确保系统资源充足
3. **模型文件**: 训练后的模型文件会保存在 `models/` 目录下
4. **数据格式**: 如果使用自定义数据集，请确保数据格式正确（payload字段和label字段）
5. **训练历史**: 训练历史数据仅在所有模型完整训练完成后才会写入数据库，避免数据重复
6. **性能监控**: 可通过监控仪表盘实时查看检测统计和趋势
7. **模型更新**: 重新训练模型后，需要重启Flask应用以加载新模型

## 数据库表结构

### detection_records
检测记录表，存储每次检测的结果

### training_data
训练数据表，存储用于训练的XSS样本

### model_info
模型信息表，存储各模型的性能指标

### training_history
训练历史表，存储模型训练过程中的loss和accuracy变化

## 技术栈

- **后端**: Flask
- **机器学习**: XGBoost, TensorFlow/Keras
- **深度学习**: BiLSTM, Transformer
- **数据处理**: Pandas, NumPy, Scikit-learn
- **数据库**: MySQL + PyMySQL
- **前端**: Bootstrap 5, Chart.js, FontAwesome

## 许可证

MIT License
