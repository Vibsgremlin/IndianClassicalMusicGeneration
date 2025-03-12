# Indian Classical Music Generator 🎵

## **Overview**
The **Indian Classical Music Generator** is a deep learning-based system that generates Indian classical music using a **Transformer model** trained on MIDI sequences. Users can select a **Raga, Tala, and Instrument** to generate new compositions. The project features an interactive UI built with **Streamlit**.

---
## **Features**
- 🎶 **Generates Indian Classical Music** based on MIDI datasets.
- 🎻 **Supports multiple instruments** like Sitar, Flute, Tabla, and Violin.
- 🎼 **Uses a Transformer Model** trained on MIDI sequences.
- 🌐 **Interactive Web UI** built with Streamlit.
- 📥 **Download generated MIDI files** directly from the UI.

---
## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/Indian-Classical-Music-Generator.git
cd Indian-Classical-Music-Generator
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### **3. Download Dataset**
Place the MIDI dataset in the project directory:
```plaintext
C:\Users\91965\maestro-v2.0.0
```
Ensure the dataset contains `.midi` or `.mid` files.

### **4. Run the Web App**
```bash
streamlit run main.py
```
This will launch an interactive web app where you can generate music.

---
## **Usage**
### **Generating Music**
1. Open the **web app**.
2. Select **Raga, Tala, and Instrument**.
3. Click **Generate Music**.
4. Download the generated **MIDI file**.

### **Convert MIDI to MP3 (Optional)**
Use **FluidSynth** and **FFmpeg**:
```bash
fluidsynth -ni soundfont.sf2 generated_music.mid -F output.wav
ffmpeg -i output.wav output.mp3
```

---
## **Project Structure**
```plaintext
📂 Indian-Classical-Music-Generator
│── main.py  # Main script with model training & UI
│── model.py  # Transformer model definition
│── preprocess.py  # Data preprocessing functions
│── requirements.txt  # Required dependencies
│── README.md  # Project documentation
│── generated_music.mid  # Sample output file
└── 📂 data/  # Contains MIDI dataset
```

---
## **Tech Stack**
- **Machine Learning**: TensorFlow, Keras
- **Music Processing**: PrettyMIDI, Librosa
- **Web App**: Streamlit
- **Data**: MIDI dataset (Maestro v2.0.0)

---
## **Contributing**
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make changes and commit: `git commit -m "Added feature"`
4. Push changes: `git push origin feature-branch`
5. Submit a Pull Request.

---
## **License**
This project is licensed under the **MIT License**.

---
## **Author**
Developed by **Your Name** – [GitHub Profile](https://github.com/your-username) 🚀

