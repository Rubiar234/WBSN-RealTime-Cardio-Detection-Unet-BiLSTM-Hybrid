# WBSN-RealTime-Cardio-Detection-Unet-BiLSTM-Hybrid LSTM for Real-Time Cardiopulmonary Detection in WBSN Data

**Pulmo-CVNet**: A hybrid deep learning framework combining Fine-tuned U-Net and Time-Weighted Multi-Task Bi-Directional LSTM (TW-MT-BiLSTM) for real-time vital sign prediction and cardiopulmonary event detection using Wireless Body Sensor Networks (WBSN).

---

## 📄 Research Paper
**Title**: Fine-Tuned U-Net and Time-Weighted Bi-Directional LSTM for Real-Time Cardiopulmonary Detection in WBSN Data

**Authors**:
- Rubia R. (Research Scholar)
- Dr. S. Sasikala (Professor & Head)

**Affiliation**: Velammal College of Engineering and Technology, Madurai, Tamil Nadu, India

---

## ✨ Key Features

- Fine-tuned U-Net for spatial feature extraction from physiological signals
- Time-Weighted Attention Mechanism to focus on clinically critical time segments
- Bi-Directional LSTM for capturing past and future temporal context
- Multi-Task Learning for simultaneous prediction of Blood Pressure, SpO₂, and Respiratory Rate
- End-to-end architecture optimized for real-time monitoring in WBSNs
- Superior performance compared to baseline U-Net + LSTM models

---

## 📊 Results Highlights

| Vital Sign       | MAE     | RMSE    | R²    | Accuracy | F1-Score |
|------------------|---------|---------|-------|----------|----------|
| Blood Pressure   | 4.12    | 5.61    | 0.92  | 60%      | 56%      |
| SpO₂             | 1.87    | 2.41    | 0.95  | **96%**  | 95%      |
| Respiratory Rate | 1.45    | 1.76    | 0.91  | 93%      | **92%**  |

**Best performing model** on MIMIC-III and PhysioNet datasets.


---

## 🛠️ Technologies Used

- Python 3.9+
- PyTorch / TensorFlow
- U-Net Architecture
- Bi-Directional LSTM
- Attention Mechanism
- Multi-Task Learning
- MIMIC-III & PhysioNet Datasets
