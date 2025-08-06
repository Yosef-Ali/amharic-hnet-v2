# 🚀 Kaggle CLI Commands for GPU Training

**Quick Reference for yosefali2**

## 🔧 One-Command Deployment

```bash
# Automated deployment (recommended)
python kaggle_auto_deploy.py
```

## 📋 Manual CLI Commands

### 1. Setup Kaggle CLI
```bash
# Install Kaggle CLI
pip install kaggle

# Verify installation
kaggle --version
```

### 2. Configure Credentials
```bash
# Your credentials are already in kaggle_credentials.json
# Copy to Kaggle directory
mkdir -p ~/.kaggle
cp kaggle_credentials.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Create & Deploy Kernel
```bash
# Create kernel (with kernel-metadata.json)
kaggle kernels push

# Check kernel status
kaggle kernels status yosefali2/amharic-gpu-training

# Download kernel output
kaggle kernels output yosefali2/amharic-gpu-training
```

### 4. Dataset Management
```bash
# Create dataset (with dataset-metadata.json)
kaggle datasets create

# Update dataset
kaggle datasets version -m "Updated training files"

# List your datasets
kaggle datasets list --user yosefali2
```

## 🎯 Quick GPU Training Deployment

### Option A: Automated Script
```bash
cd kaggle_gpu_production
python kaggle_auto_deploy.py
# Follow prompts - fully automated!
```

### Option B: Manual Steps
```bash
# 1. Create training script (main.py)
# 2. Create kernel config (kernel-metadata.json)
# 3. Deploy
kaggle kernels push
```

## 📊 Monitoring Commands

```bash
# Check kernel status
kaggle kernels status yosefali2/amharic-gpu-training

# View kernel logs (if available)
kaggle kernels output yosefali2/amharic-gpu-training --path ./output

# List your kernels
kaggle kernels list --user yosefali2
```

## 📥 Download Trained Models

```bash
# Download all kernel outputs
kaggle kernels output yosefali2/amharic-gpu-training --path ./models

# Your trained models will be in:
# ./models/best_model.pt
# ./models/final_model.pt
```

## 🔗 Useful URLs

- **Your Kernels**: https://www.kaggle.com/yosefali2/code
- **Your Datasets**: https://www.kaggle.com/yosefali2/datasets
- **GPU Training Kernel**: https://www.kaggle.com/code/yosefali2/amharic-gpu-training

## ⚡ Pro Tips

1. **Always enable GPU**: Set `"enable_gpu": true` in kernel-metadata.json
2. **Enable Internet**: Set `"enable_internet": true` for package installation
3. **Monitor training**: Check kernel status every 30 minutes
4. **Download models**: Save trained models before kernel expires
5. **Use private kernels**: Set `"is_private": true` for sensitive work

## 🚨 Troubleshooting

### Common Issues
```bash
# API key not found
echo $KAGGLE_USERNAME  # Should show: yosefali2
echo $KAGGLE_KEY       # Should show your key

# Or check file
cat ~/.kaggle/kaggle.json

# Permission denied
chmod 600 ~/.kaggle/kaggle.json

# Kernel already exists
kaggle kernels push --id yosefali2/amharic-gpu-training
```

### GPU Not Available
- Check kernel settings: enable_gpu = true
- Verify GPU quota in Kaggle account
- Try different kernel type (script vs notebook)

## 📈 Expected Results

```bash
# Successful deployment shows:
✅ Kernel deployed successfully!
🔗 Kernel URL: https://www.kaggle.com/code/yosefali2/amharic-gpu-training

# Training output shows:
GPU Available: True
GPU: Tesla P100-PCIE-16GB
Model parameters: 19,234,567
Epoch 1/20: Average Loss = 4.2341
💾 Best model saved! Loss: 3.8456
🏆 Training Complete!
```

## 🎉 Success Checklist

- [ ] Kaggle CLI installed
- [ ] Credentials configured
- [ ] Kernel deployed
- [ ] GPU enabled
- [ ] Training started
- [ ] Models downloaded
- [ ] Ready for competition!

---

**Ready to deploy? Run: `python kaggle_auto_deploy.py`** 🚀