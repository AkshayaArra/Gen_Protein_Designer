"""
GenProteinDesigner - SAFE DEMO

This Streamlit app is a fully self-contained, safe demonstration of the
GenProteinDesigner scaffold. It does NOT generate or optimize real biological
sequences. Instead it creates randomized placeholder amino-acid sequences
and computes non-actionable summary statistics for presentation / teaching.

Features:
- Prompt input with simple safety check (keyword blocklist)
- Generate randomized placeholder peptide/protein sequences (length inferred from prompt)
- Compute simple, non-actionable features (length, avg hydrophobicity proxy, net charge proxy,
  simple helix/sheet fraction proxies, AA composition)
- Display interactive table, plots, and allow CSV / FASTA download
- Clear safety & ethics banner

Run:
    pip install streamlit pandas numpy altair
    streamlit run app.py
"""

from typing import List, Dict
import streamlit as st
import random
import re
import pandas as pd
from collections import Counter
from datetime import datetime

# -----------------------
# Safety / Utils
# -----------------------
DISALLOWED_KEYWORDS = [
    "toxin", "increase virulence", "kill", "weaponize", "bioweapon",
    "pathogen", "release", "harmful", "biological agent", "weapon",
    "enhance transmiss", "grow bacteria", "grow virus"
]

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids

KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "D": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5
}

HELIX_AA = set("AKLHERMQ")  # coarse proxy
SHEET_AA = set("VICYFTW")   # coarse proxy


def check_prompt_safe(prompt: str) -> tuple:
    """Very simple safety filter — blocks prompts containing dangerous keywords."""
    p = (prompt or "").lower()
    for kw in DISALLOWED_KEYWORDS:
        if kw in p:
            return False, f"Disallowed keyword detected: '{kw}'"
    if re.search(r"(increase|enhanc(e|ing)|boost).*(virul|transm|toxin)", p):
        return False, "Request appears to ask for enhancing harmful properties"
    return True, "ok"


def parse_requested_length(prompt: str, default: int = 40) -> int:
    """Try to parse desired length from prompt text, otherwise return default."""
    if not prompt:
        return default
    m = re.search(r"(\d{1,3})\s*(aa|residue|residues|length|aa\.)", prompt.lower())
    if m:
        try:
            val = int(m.group(1))
            if 6 <= val <= 200:
                return val
        except:
            pass
    return default


# -----------------------
# Safe generator (placeholder)
# -----------------------
def _random_peptide(length: int) -> str:
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def safe_generate_sequences(prompt: str, n: int = 5) -> List[str]:
    """
    Generate randomized placeholder sequences based on a prompt.

    NOTE: This is intentionally a non-actionable placeholder generator.
    It does NOT optimize or design sequences in any biological sense.
    """
    length = parse_requested_length(prompt, default=40)
    seqs = [_random_peptide(length) for _ in range(n)]
    return seqs


# -----------------------
# Scoring / features (non-actionable)
# -----------------------
def compute_sequence_features(seq: str) -> Dict:
    seq = seq.strip().upper()
    L = len(seq)
    counts = Counter(seq)
    aa_frac = {aa: counts.get(aa, 0) / L for aa in AMINO_ACIDS}

    hyd_sum = 0.0
    for aa, ch in counts.items():
        hyd_sum += KD_SCALE.get(aa, 0.0) * ch
    avg_hydrophobicity = hyd_sum / L if L > 0 else 0.0

    pos = counts.get("K", 0) + counts.get("R", 0) + counts.get("H", 0)
    neg = counts.get("D", 0) + counts.get("E", 0)
    net_charge = pos - neg

    helix_count = sum(counts.get(aa, 0) for aa in HELIX_AA)
    sheet_count = sum(counts.get(aa, 0) for aa in SHEET_AA)
    helix_frac = helix_count / L if L > 0 else 0.0
    sheet_frac = sheet_count / L if L > 0 else 0.0

    feats = {
        "length": int(L),
        "avg_hydrophobicity": float(round(avg_hydrophobicity, 3)),
        "net_charge_proxy": int(net_charge),
        "helix_frac_proxy": float(round(helix_frac, 3)),
        "sheet_frac_proxy": float(round(sheet_frac, 3)),
    }
    # Add fractions as native Python floats
    for aa in AMINO_ACIDS:
        feats[f"frac_{aa}"] = float(round(aa_frac[aa], 3))
    return feats


# -----------------------
# Helpers for downloads
# -----------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fasta_from_seqs(seqs: List[str], ids: List[str] = None) -> str:
    if ids is None:
        ids = [f"seq_{i+1}" for i in range(len(seqs))]
    lines = []
    for i, s in enumerate(seqs):
        lines.append(f">{ids[i]}")
        lines.append(s)
    return "\n".join(lines)


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="GenProteinDesigner — SAFE Demo", layout="centered")
st.title("GenProteinDesigner — SAFE Demo")
st.markdown(
    """
**Warning & scope:** This is a *safe demonstration* scaffold only.  
It **does not** generate or optimize real functional biological sequences.  
The generator produces randomized placeholder amino-acid sequences for teaching and UI testing.
"""
)

# Sidebar controls
st.sidebar.header("Design parameters")
prompt = st.sidebar.text_area("Design prompt (for demo)", value="Design a 25 aa cationic amphipathic peptide")
n = st.sidebar.number_input("Number of candidates", min_value=1, max_value=50, value=6, step=1)
seed = st.sidebar.number_input("Random seed (for reproducibility)", min_value=0, max_value=2147483647, value=42)
st.sidebar.markdown("---")
st.sidebar.markdown("**Downloads**")
st.sidebar.markdown("- CSV of results\n- FASTA of sequences")
st.sidebar.markdown("---")
st.sidebar.markdown("**Safety**\nThis demo blocks dangerous prompts. For real design work, follow institutional approvals.")

# Generate button
st.write("### Input")
st.write("Prompt (demo):", f"_{prompt}_")

if st.button("Run demo generation"):
    random.seed(seed)
    ok, reason = check_prompt_safe(prompt)
    if not ok:
        st.error(f"Prompt rejected: {reason}")
    else:
        with st.spinner("Generating placeholder sequences..."):
            seqs = safe_generate_sequences(prompt, n=int(n))
            # compute features - build dict list carefully
            rows = []
            for i, s in enumerate(seqs, start=1):
                feats = compute_sequence_features(s)
                # Create row with explicit Python types
                row = {
                    "id": f"cand_{i}", 
                    "sequence": str(s)
                }
                row.update(feats)
                rows.append(row)
            
            # Create DataFrame with explicit column order
            df = pd.DataFrame(rows)

        st.success("Generation complete (placeholder sequences).")
        st.write("### Candidate sequences (placeholders)")
        
        # Display subset of columns
        display_cols = ["id", "sequence", "length", "avg_hydrophobicity", "net_charge_proxy", "helix_frac_proxy", "sheet_frac_proxy"]
        st.dataframe(df[display_cols])

        # Plots
        st.write("### Summary plots")
        try:
            import altair as alt
            # histogram of avg_hydrophobicity
            chart1 = alt.Chart(df).mark_bar().encode(
                alt.X("avg_hydrophobicity:Q", bin=alt.Bin(maxbins=20)),
                y='count()'
            ).properties(title="Avg hydrophobicity distribution")
            st.altair_chart(chart1, use_container_width=True)

            # net charge
            chart2 = alt.Chart(df).mark_bar().encode(
                alt.X("net_charge_proxy:Q"),
                y='count()'
            ).properties(title="Net charge proxy")
            st.altair_chart(chart2, use_container_width=True)

        except Exception as e:
            # fallback simple display
            st.write("**Hydrophobicity values:**")
            st.write(list(df["avg_hydrophobicity"]))
            st.write("**Net charge proxies:**")
            st.write(list(df["net_charge_proxy"]))

        # Download buttons
        csv_bytes = df_to_csv_bytes(df)
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"genprot_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        fasta_txt = fasta_from_seqs(df["sequence"].tolist(), ids=df["id"].tolist())
        st.download_button(
            "Download FASTA",
            data=fasta_txt,
            file_name=f"genprot_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fasta",
            mime="text/plain"
        )

        # Show a detailed view for selected candidate
        st.write("### Inspect a candidate")
        sel = st.selectbox("Select candidate", df["id"].tolist())
        if sel:
            selrow = df[df["id"] == sel].iloc[0].to_dict()
            st.write("**Sequence:**", selrow["sequence"])
            st.write("**Detailed features:**")
            feature_dict = {k: v for k, v in selrow.items() if k not in ["id", "sequence"]}
            st.json(feature_dict)

# Footer - educational resources
st.markdown("---")
st.markdown(
    """
**Educational notes:**  
- This app is intended for teaching system design and UI for GenAI-driven bioinformatics pipelines.  
- It intentionally avoids any actionable sequence-design capability.  
- If you require a version that integrates legitimate sequence-generation models for institutional research, please obtain formal biosafety approvals and replace the `safe_generate_sequences` stub with approved modules under supervision.
"""
)