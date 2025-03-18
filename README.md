# 📊 Parquet Viewer with GMM Visualization  

🚀 **A powerful and interactive Streamlit-based tool** for visualizing **log data** and **Gaussian Mixture Models (GMMs)**. This tool supports **frequency (f)**, **pulsewidth (w)**, and **pulse repetition interval (p)** analysis, allowing users to explore and compare raw measurements with probabilistic models.

## 🎯 Features  
✔ **Interactive Parquet File Viewer** – Load and explore **GMM and LOG parquet files** dynamically.  
✔ **Advanced GMM Visualization** – Overlay **Gaussian Mixture Model (GMM) components** onto frequency and pulsewidth distributions.  
✔ **Multi-Feature Support** – Analyze **frequency (f)**, **pulsewidth (w)**, and **PRI (p)** with dedicated plots.  
✔ **Row Selection & Highlighting** – Select a row using a slider **or keyboard shortcuts (`M` / `N`)**, and highlight it in the table.  
✔ **Scrollable 7-Row View** – Always display **3 rows before and after** the selected row, **keeping the selected row centered**.  

---

## 🖥️ Demo  
🔹 **Select a row from the slider** to visualize data and corresponding GMM models.  
🔹 **Use keyboard shortcuts (`M` / `N`)** to navigate between rows efficiently.  
🔹 **Switch between tabs** to view different feature analyses (`f`, `w`, `p`).  

![Demo Image](https://github.com/hugodrak/ml_viewer_streamlit/blob/main/demo_image.png?raw=true)

---

## 📦 Installation  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/yourusername/parquet-viewer.git
cd parquet-viewer
```

2️⃣ **Create and activate a virtual environment**  
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

3️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit app**  
```bash
streamlit run pq_viewer.py
```

---

## 📂 File Structure  
```
parquet-viewer/
│── pq_viewer.py                # Main Streamlit application
│── requirements.txt      # List of dependencies
│── README.md             # Project documentation (this file)
```


---

## 🎮 Keyboard Shortcuts  
| Key  | Action                 |
|------|------------------------|
| `N`  | Move to **next row**   |
| `M`  | Move to **previous row** |

---

## 🔬 How It Works  

### **1️⃣ Data Loading**  
- The user uploads **two parquet files**:
  - **LOG file:** Contains raw measurement data.
  - **GMM file:** Contains Gaussian Mixture Model parameters.

### **2️⃣ Row Selection & Highlighting**  
- Users select a row via a **slider** or **keyboard shortcuts (`M` / `N`)**.  
- The table always **displays 7 rows**, ensuring the **selected row stays centered** (except at the dataset’s start or end).

### **3️⃣ Visualization**  
- **Frequency (f) and Pulsewidth (w):**  
  - Plots normal distributions based on **mean & stddev values**.  
  - Overlays **GMM components** as dashed curves.  
- **PRI (p):**  
  - Displays **raw measured data points** without curve fitting.  
  - Overlays **GMM priors** for comparison.  

---

## 🚀 Future Improvements  
✅ **Support for more data formats (CSV, JSON, HDF5)**  
✅ **Customizable GMM fitting parameters**  
✅ **Export visualizations as images/PDFs**  
✅ **Additional analytics and statistics**  

---

## 👨‍💻 Contributing  
Contributions are welcome! Feel free to submit **issues** and **pull requests**.

1. **Fork the repository**  
2. **Create a feature branch**  
3. **Commit your changes**  
4. **Push and open a PR**  

---

## 📝 License  
This project is **open-source** under the **MIT License**.

---

💡 **Enjoy using the Parquet Viewer!** Let us know if you have any feedback. 🚀✨

---