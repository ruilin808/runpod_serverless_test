# Unsloth Vision Model Training Scripts Comparison

## Creation Sequence Analysis

Based on the code evolution and complexity, the likely sequence is:

**1. uinit.py** → **2. uinit1.py** → **3. utest.py** → **4. utest_noTEDS.py** → **5. utest_balanced.py** → **6. utest_tables.py** → **7. utest_tables_updated.py** → **8. utest_tables_updated1.py**

---

## Detailed Comparison Table

| Feature | uinit.py | uinit1.py | utest.py | utest_noTEDS.py | utest_balanced.py | utest_tables.py | utest_tables_updated.py | utest_tables_updated1.py |
|---------|----------|-----------|----------|-----------------|-------------------|-----------------|-------------------------|---------------------------|
| **Status** | ✅ Original | ✅ Enhanced | ✅ Advanced | ✅ Simplified | ✅ Production | ✅ Batch Processing | ✅ Filtered + Batch | ✅ Final Version |
| **LoRA Rank (r)** | 16 | 16 | 8 | 8 | 4 | 8 | 4 | 4 |
| **LoRA Alpha** | 16 | 16 | 8 | 8 | 8 | 8 | 8 | 8 |
| **LoRA Dropout** | 0 | 0 | 0 | 0 | 0.1 | 0 | 0.1 | 0.1 |
| **Dataset Split** | Train only | Train + Val | Train + Val | Train + Val | Train + Val | Train + Val | Train + Val | Train + Val |
| **HTML Filtering** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Advanced | ❌ None | ✅ Advanced | ✅ Advanced |
| **HTML Cleaning** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Yes | ✅ Basic | ✅ Advanced | ✅ Advanced |
| **TEDS Metrics** | ❌ None | ✅ Yes | ✅ Yes | ❌ Disabled | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Training Callback** | ❌ None | ✅ Yes | ✅ Yes | ❌ Commented | ❌ None | ❌ None | ❌ None | ❌ None |
| **Batch Size** | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 1 |
| **Gradient Accumulation** | 4 | 4 | 8 | 4 | 8 | 4 | 8 | 8 |
| **Learning Rate** | 2e-4 | 2e-4 | 1e-4 | 5e-5 | 2e-5 | 5e-5 | 2e-5 | 2e-5 |
| **Max Steps** | 30 | 30 | 30 | - | - | - | - | - |
| **Epochs** | - | - | 5 | 3 | 3 | 3 | 3 | 3 |
| **Warmup** | 5 steps | 5 steps | 10 steps | 20 steps | 0.1 ratio | 20 steps | 0.1 ratio | 0.1 ratio |
| **Weight Decay** | 0.01 | 0.01 | 0.1 | 0.01 | 0.1 | 0.01 | 0.1 | 0.1 |
| **LR Scheduler** | linear | linear | cosine | cosine | cosine | cosine | cosine | cosine |
| **Instruction** | Basic | Basic | Basic | Basic | Detailed | Basic | Detailed | Detailed |
| **Pre-training Inference** | ❌ None | ❌ None | ✅ Yes | ✅ Yes | ✅ On samples | ✅ Batch folder | ✅ On samples | ✅ On samples |
| **Post-training Inference** | ❌ None | ❌ None | ✅ Yes | ✅ Yes | ✅ On samples | ✅ Batch folder | ✅ On samples | ✅ On samples |
| **Sample Preparation** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Top 20 by length | ❌ None | ❌ None | ✅ Top 20 by length |
| **Output Folders** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Multiple | ✅ Multiple | ✅ Multiple | ✅ Multiple |
| **File Management** | ❌ None | ❌ None | ❌ None | ❌ None | ✅ Images/HTML | ❌ None | ❌ None | ✅ Images/HTML |
| **Final Evaluation** | ❌ None | ✅ 10 samples | ✅ 3 samples | ❌ Commented | ❌ None | ❌ None | ❌ None | ❌ None |
| **Save Strategy** | epoch | epoch | epoch | epoch | steps | epoch | steps | steps |
| **Eval Strategy** | - | - | epoch | epoch | steps | epoch | steps | steps |

---

## Key Evolution Points

### **uinit.py (Original)**
- Basic foundation script
- Single training dataset
- Simple configuration
- No evaluation metrics

### **uinit1.py (First Enhancement)**
- Added validation dataset
- Introduced TEDS metrics
- Added evaluation callback
- Still basic configuration

### **utest.py (Advanced Testing)**
- Reduced LoRA parameters for stability
- Added comprehensive inference testing
- More conservative training settings
- Better evaluation framework

### **utest_noTEDS.py (Simplified)**
- Disabled TEDS for troubleshooting
- Adjusted learning rates
- Streamlined for basic functionality
- Removed evaluation complexity

### **utest_balanced.py (Production Ready)**
- Advanced HTML filtering and cleaning
- Sample preparation and management
- Comprehensive file organization
- Production-quality features

### **utest_tables.py (Batch Processing)**
- Added batch inference on folders
- Basic HTML cleaning
- Focused on processing multiple files
- Removed sample preparation

### **utest_tables_updated.py (Filtered Batch)**
- Combined HTML filtering with batch processing
- Advanced cleaning functions
- Better error handling
- Improved file management

### **utest_tables_updated1.py (Final Version)**
- Complete feature set
- Sample preparation + filtering + batch processing
- Comprehensive file organization
- Production-ready with all optimizations

---

## Recommendations

**Use utest_tables_updated1.py for production** - it combines all the best features:
- Advanced HTML filtering and cleaning
- Sample preparation and organization
- Comprehensive inference testing
- Optimal training parameters
- Complete file management system