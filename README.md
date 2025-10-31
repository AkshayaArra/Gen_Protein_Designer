# Gen_Protein_Designer

# 🧬 GenProteinDesigner — SAFE Demo

This project is an **educational Generative AI simulation** for protein sequence design — built with **Streamlit**.

✅ Fully safe  
✅ No biological functionality  
✅ Generates **random placeholder peptides**  
✅ Computes simple statistics (length, hydrophobicity, charge)  
✅ Perfect for AI + Bioinformatics learning  

---

## 🚀 Live Demo Features

| Feature | Description |
|---|---|
🧠 Prompt input | User enters a "design objective"  
🔒 Safety filter | Blocks prohibited biological terms  
🧬 Random peptide generator | Creates placeholder sequences (20 amino acids)  
📊 Property analysis | Hydrophobicity, charge, helix/sheet tendencies  
📁 Export | CSV + FASTA  
📈 Visuals | Interactive plots (Altair)  

This is **NOT a real design model** — it's a **safe UI and logic demonstration only**.

---

## 🎯 Purpose

- Teach AI-driven biomedical software design
- Demonstrate bioinformatics UI/UX
- Safe alternative to running real bio-LLMs (ProtGPT, ESM-2, AlphaFold, etc.)
- Ideal for **classroom use, projects, and student research**

---

## 🔐 Safety

This project actively prevents misuse:

- Filters dangerous biological prompts
- Does **not** generate biologically functional sequences
- Includes safety notices & ethical disclaimers
- Intended for **education only**

---

## 🛠️ Tech Stack

- Streamlit UI
- Python 3.10+
- pandas, numpy, altair
- Hugging Face Spaces deployment

---

## ▶️ Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
