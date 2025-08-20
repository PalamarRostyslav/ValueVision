# ValueVision 🎯

**AI-Powered Price Prediction System for Amazon Products**

A comprehensive end-to-end machine learning platform that processes Amazon product data, trains AI models, and provides intelligent price predictions. 
ValueVision combines advanced data processing, multiple AI fine-tuning strategies, and intelligent model management.

## 🌟 Overview

ValueVision is a complete ML/AI platform that transforms raw Amazon product data into intelligent price prediction models. It features:

- **🔄 Intelligent Data Pipeline**: Processes millions of Amazon products with smart sampling and balancing
- **🤖 Multiple AI Strategies**: OpenAI GPT fine-tuning, Random Forest, feature-based models, and more
- **💾 Smart Model Management**: Automatic model persistence, reuse, and cost optimization
- **📊 Advanced Analytics**: Comprehensive visualizations and performance metrics
- **⚡ Production Ready**: Scalable architecture with multi-processing and optimization features

  ## 🏗️ Architecture & Project Structure

```
ValueVision/
├── 🎯 CORE PIPELINE
│   ├── main.py                   # Main orchestrator with 15+ command options
│   ├── config/
│   │   └── settings.py           # Centralized configuration with .env support
│   └── .env.example             # Environment variables template
│
├── 📊 DATA PROCESSING ENGINE
│   ├── src/data/
│   │   ├── pipeline.py           # Smart data processing pipeline
│   │   ├── loaders.py            # Multi-format dataset loaders
│   │   ├── sampling.py           # Intelligent sampling algorithms
│   │   ├── dataset_creator.py    # Multi-format export (Pickle, HuggingFace)
│   │   └── models.py             # Data models and schemas
│   │
│   ├── src/visualization/
│   │   └── visualizer.py         # Advanced analytics and plotting
│   │
│   └── src/utils/
│       ├── environment.py        # Environment setup and authentication
│       └── testing.py            # Testing utilities
│
├── 🤖 AI & MACHINE LEARNING
│   ├── src/ai_models/
│   │   ├── base.py               # Abstract base for AI models
│   │   ├── openai_model.py       # OpenAI GPT fine-tuning implementation
│   │   └── model_manager.py      # Smart model persistence & reuse system
│   │
│   ├── src/testing/
│   │   ├── tester.py             # Model evaluation framework
│   │   ├── fine_tuning.py        # Fine-tuning base classes
│   │   └── strategies/           # Multiple AI training strategies
│   │       ├── openai_strategy.py       # OpenAI GPT fine-tuning
│   │       ├── random_forest.py         # Random Forest + Word2Vec
│   │       ├── feature_based.py         # Feature engineering approach
│   │       └── random_seed.py           # Seed optimization
│   │
├── 📁 OUTPUT & DATA
│   ├── output/
│   │   ├── models/              # AI model registry and metadata
│   │   ├── fine_tuning/         # Training files and logs
│   │   ├── train.pkl            # Processed training dataset
│   │   ├── test.pkl             # Processed test dataset
│   │   └── dataset/             # HuggingFace format exports
│   │
└── 🔧 UTILITIES & COMPATIBILITY
    ├── items.py                 # Backward compatibility for pickle loading
    ├── data_viewer.py          # Legacy data viewer (maintained)
    ├── test_*.py               # Comprehensive test suites
    └── requirements.txt         # Python dependencies
```


## 🚀 Core Features

### 📊 **Data Processing Engine**

- **Multi-Source Loading**: Process 5+ Amazon product categories simultaneously
- **Intelligent Sampling**: Price-based and category-balanced sampling algorithms
- **Smart Data Balancing**: Ensures representative datasets across price ranges
- **Multi-Format Export**: Pickle, HuggingFace datasets, JSON formats
- **Memory Optimization**: Efficient processing of millions of products
- **Parallel Processing**: Multi-core optimization for large datasets

### 🤖 **AI & Machine Learning**

- **OpenAI GPT Fine-Tuning**: State-of-the-art language model fine-tuning for price prediction
- **Multiple AI Strategies**: Random Forest, Feature Engineering, Seed Optimization
- **Smart Model Management**: Automatic model persistence, reuse, and cost optimization
- **Performance Analytics**: Comprehensive model evaluation and metrics
- **Hyperparameter Optimization**: Automated parameter tuning
- **Cross-Validation**: Robust model validation techniques

### 💾 **Model Management System**

- **🔄 Automatic Model Reuse**: Prevents costly retraining of identical models
- **💰 Cost Optimization**: Save money by reusing existing fine-tuned models
- **📋 Model Registry**: Track all models with metadata, metrics, and versioning
- **⚡ Instant Deployment**: Use specific models immediately with model ID specification
- **📈 Performance Tracking**: Monitor model usage and effectiveness over time

### 📊 **Advanced Analytics**

- **Rich Visualizations**: Price distributions, category analysis, correlation matrices
- **Performance Metrics**: RMSE, MAE, accuracy scores, error analysis
- **Data Quality Insights**: Missing data analysis, outlier detection
- **Training Progress**: Real-time monitoring of model training
- **Comparative Analysis**: Side-by-side model performance comparison

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI API access (for GPT fine-tuning)
- HuggingFace account (for dataset access)
- 8GB+ RAM recommended for large datasets

### Quick Installation

1. **Clone the repository:**

```bash
git clone https://github.com/PalamarRostyslav/ValueVision.git
cd ValueVision
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
# Required for AI fine-tuning:
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

4. **Verify installation:**

```bash
python main.py --help
```


## 🎯 Usage Guide

### 🚀 Quick Start Commands

```bash
# 🔥 FASTEST: Use existing processed data
python main.py --use-existing

# 📊 ANALYSIS: Quick data exploration
python main.py --analysis-only

# 🔄 FULL: Complete pipeline from scratch
python main.py

# 🎯 SPECIFIC: Process only certain categories
python main.py --datasets Electronics Automotive Home
```

### 🤖 AI Fine-Tuning & Model Training

#### OpenAI GPT Fine-Tuning

```bash
# 🆕 Train new OpenAI model (reuses existing if identical data)
python main.py --fine-tune openai

# 💰 Force new training (ignores existing models)
python main.py --fine-tune openai --force-new-model

# ⚡ Use specific pre-trained model instantly
python main.py --fine-tune openai --model-id ft:gpt-4o-mini:your-org:model:abc123
```

#### Alternative AI Strategies

```bash
# 🌲 Random Forest with Word2Vec
python main.py --fine-tune random-forest

# ⚙️ Feature-based machine learning
python main.py --fine-tune feature-based

# 🎲 Random seed optimization
python main.py --fine-tune random-seed
```

### 💾 Model Management

#### View & Manage Models

```bash
# 📋 List all saved models with details
python main.py --list-models

# 🧹 Clean up old models (keeps latest 3 per provider)
python main.py --cleanup-models 30  # Remove models older than 30 days

# 📊 Test existing models
python main.py --test --test-size 100
```

#### Cost-Saving Model Reuse

```bash
# First run: Trains and saves model
python main.py --fine-tune openai
# Output: "✓ Model saved: ft:gpt-4o-mini:org:model:abc123"

# Subsequent runs: Instant reuse (no cost!)
python main.py --fine-tune openai
# Output: "✓ Found existing model, reusing..."

# Manual model specification
python main.py --fine-tune openai --model-id ft:gpt-4o-mini:org:model:abc123
```

### 🔧 Advanced Usage

#### Custom Data Processing

```bash
# Force reload specific datasets
python main.py --datasets Electronics --force-reload

# Use existing data but force new analysis
python main.py --use-existing --analysis-only

# Custom test size for model evaluation
python main.py --test --test-size 500
```

#### Development Workflow

```bash
# 1. Initial setup (full pipeline)
python main.py

# 2. Model experimentation (fast iterations)
python main.py --use-existing --fine-tune openai

# 3. Production deployment
python main.py --fine-tune openai --model-id your_best_model_id
```

## 📊 Complete Command Reference

| Command                     | Description                     | Example                                                               |
| --------------------------- | ------------------------------- | --------------------------------------------------------------------- |
| **Data Processing**         |
| `python main.py`            | Full pipeline from scratch      | `python main.py`                                                      |
| `--use-existing`            | Use cached data if available    | `python main.py --use-existing`                                       |
| `--analysis-only`           | Quick data analysis only        | `python main.py --analysis-only`                                      |
| `--force-reload`            | Force reload even with existing | `python main.py --force-reload`                                       |
| `--datasets LIST`           | Process specific categories     | `python main.py --datasets Electronics Home`                          |
| **AI Fine-Tuning**          |
| `--fine-tune openai`        | OpenAI GPT fine-tuning          | `python main.py --fine-tune openai`                                   |
| `--fine-tune random-forest` | Random Forest training          | `python main.py --fine-tune random-forest`                            |
| `--fine-tune feature-based` | Feature engineering approach    | `python main.py --fine-tune feature-based`                            |
| `--fine-tune random-seed`   | Seed optimization               | `python main.py --fine-tune random-seed`                              |
| **Model Management**        |
| `--model-id ID`             | Use specific model              | `python main.py --fine-tune openai --model-id ft:gpt-4o-mini:org:abc` |
| `--force-new-model`         | Force new training              | `python main.py --fine-tune openai --force-new-model`                 |
| `--list-models`             | Show all saved models           | `python main.py --list-models`                                        |
| `--cleanup-models DAYS`     | Remove old models               | `python main.py --cleanup-models 30`                                  |
| **Testing & Evaluation**    |
| `--test`                    | Run model testing               | `python main.py --test`                                               |
| `--test-size N`             | Custom test size                | `python main.py --test --test-size 500`                               |

## 💰 Cost Optimization Features

### Smart Model Reuse

- **Automatic Detection**: Identifies identical training configurations
- **Zero-Cost Reuse**: Skip expensive retraining for same data
- **Model Registry**: Track and version all trained models
- **Intelligent Matching**: Hash-based training data comparison

## 🔧 Advanced Configuration

### Core Settings (`config/settings.py`)

```python
# Data Processing
DATASET_NAMES = ['Electronics', 'Automotive', 'Home', ...]  # Categories to process
TRAIN_SIZE = 400000                    # Training dataset size
TEST_SIZE = 2000                      # Test dataset size
MAX_ITEMS_PER_PRICE = 1200           # Price point balancing
RANDOM_SEED = 42                     # Reproducibility

# AI Model Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # Auto-loaded from .env
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"          # Default OpenAI model
HF_TOKEN = os.getenv("HF_TOKEN")                  # HuggingFace access

# Advanced Processing
USE_MULTIPROCESSING = True           # Enable parallel processing
MEMORY_OPTIMIZATION = True           # Optimize for large datasets
VERBOSE_LOGGING = True              # Detailed operation logs
```

### Environment Variables (`.env`)

```bash
# Required for AI fine-tuning
OPENAI_API_KEY=sk-your-openai-key
HF_TOKEN=hf_your-huggingface-token

# Optional: Custom model preferences
OPENAI_MODEL=gpt-4o-mini-2024-07-18
DEFAULT_TEST_SIZE=250
```

### Programmatic Usage

```python
# Import core modules
from src.data.pipeline import DataPipeline
from src.ai_models.openai_model import OpenAIFineTuning
from src.ai_models.model_manager import ModelManager

# Data Processing
pipeline = DataPipeline(use_existing_data=True)
train_data, test_data = pipeline.load_processed_data()

# AI Model Training
openai_model = OpenAIFineTuning(model_name="gpt-4o-mini-2024-07-18")
predictor = openai_model.fine_tune_and_create_predictor(train_data, test_data)

# Model Management
manager = ModelManager()
models = manager.list_models(provider="openai")
existing_model = manager.find_existing_model("openai", "gpt-4o-mini", training_info)

# Make Predictions
price_prediction = predictor(item)
```

## 🔄 Data Processing Pipeline

### Pipeline Stages

1. **🔧 Environment Setup**

   - Load environment variables and API keys
   - Authenticate with HuggingFace and OpenAI
   - Initialize logging and configuration

2. **📥 Multi-Source Data Loading**

   - Download Amazon product datasets (10+ categories)
   - Handle various data formats and schemas
   - Memory-efficient streaming for large datasets

3. **🔍 Data Analysis & Validation**

   - Generate overview statistics and visualizations
   - Identify data quality issues and outliers
   - Create price distribution and category analysis

4. **⚖️ Intelligent Sampling**

   - Price-based balancing to prevent bias
   - Category-weighted sampling for representation
   - Outlier handling and data cleaning

5. **✂️ Train/Test Splitting**

   - Stratified splitting to maintain distributions
   - Configurable split ratios and sizes
   - Reproducible splits with seed control

6. **💾 Multi-Format Export**

   - Pickle files for Python compatibility
   - HuggingFace datasets for ML frameworks
   - JSON exports for external tools

7. **📊 Final Analytics**
   - Generate comprehensive data reports
   - Performance metrics and statistics
   - Visual summaries and insights

### Data Flow Architecture

```
Raw Amazon Data → Loading → Validation → Sampling → Splitting → Export
                     ↓         ↓          ↓         ↓         ↓
                 Analytics  Quality   Balancing  Testing   Multi-Format
                           Checks               Validation    Output
```

## 🤖 AI Fine-Tuning Strategies

### 1. 🎯 **OpenAI GPT Fine-Tuning** (Recommended)

- **Technology**: GPT-4o-mini fine-tuning via OpenAI API
- **Advantages**: State-of-the-art accuracy, handles complex product descriptions
- **Use Case**: Production-grade price prediction with natural language understanding
- **Cost**: ~$1-5 per training job (with smart reuse system)
- 
<img width="1197" height="861" alt="Pasted image 20250820203345" src="https://github.com/user-attachments/assets/47934cd4-4c9d-4eb2-b9d6-6b9fb0abf1c9" />

### 2. 🌲 **Random Forest + Word2Vec**

- **Technology**: Ensemble learning with semantic embeddings
- **Advantages**: Fast training, interpretable features, good baseline
- **Use Case**: Quick prototyping and feature importance analysis
- **Cost**: Free, uses local computation

<img width="1197" height="859" alt="Pasted image 20250820190405" src="https://github.com/user-attachments/assets/a69c7832-0dc9-4606-a155-aa639ba4dbc8" />


### 3. ⚙️ **Feature-Based ML**

- **Technology**: Traditional ML with engineered features
- **Advantages**: Highly interpretable, fast inference, low resource usage
- **Use Case**: Scenarios requiring model explainability
- **Cost**: Free, minimal computational requirements
- 
<img width="1199" height="860" alt="Pasted image 20250820183635" src="https://github.com/user-attachments/assets/defffa4a-39d4-464c-b05d-f63cfc97f888" />

### 4. 🎲 **Random Seed Optimization**

- **Technology**: Hyperparameter optimization through seed tuning
- **Advantages**: Improves any model's performance, no additional complexity
- **Use Case**: Maximizing performance of existing models
- **Cost**: Free, automated optimization
- 
<img width="1199" height="864" alt="Pasted image 20250820183656" src="https://github.com/user-attachments/assets/8812ab10-ef56-481c-b950-820462f9aa27" />

### Performance Comparison

| Strategy          | Accuracy     | Speed      | Cost     | Interpretability | Best For   |
| ----------------- | -----------  | ---------- | -------  | ---------------- | ---------- |
| OpenAI GPT        | 🟢 Highest  | 🟡 Medium  | 🟡 Paid | 🔴 Low           | Production |
| Random Forest     | 🟡 Good     | 🟢 Fast    | 🟢 Free | 🟢 High          | Baseline   |
| Feature-Based     | 🔴 Low      | 🟢 Fastest | 🟢 Free | 🟢 Highest       | Analysis   |
| Seed Optimization | 🔴 Low      | 🟢 Fast    | 🟢 Free | 🟢 High          | Tuning     |

### Performance Tips

- **💾 Use `--use-existing`** for faster iterations during development
- **🎯 Start with smaller datasets** to validate workflows
- **⚡ Use specific model IDs** for production to avoid retraining



