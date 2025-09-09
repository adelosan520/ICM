import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import umap
except Exception:
    umap = None

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT_DIR  = os.environ.get("OUT_DIR", "./out")

counts_path = os.path.join(DATA_DIR, "Human_Embryo_Counts.csv")
meta_path   = os.path.join(DATA_DIR, "Human_Sample_Info.csv")
genes_saved = os.path.join(DATA_DIR, "Saved_cESFW_Genes.npy")

plots_dir = os.path.join(OUT_DIR, "Plots")
os.makedirs(plots_dir, exist_ok=True)

counts = pd.read_csv(counts_path, index_col=0)
sel_genes = list(np.load(genes_saved, allow_pickle=True))
keep = [g for g in sel_genes if g in counts.columns]
if not keep:
    raise SystemExit("No overlap between Saved_cESFW_Genes and counts columns.")
X = counts.loc[:, keep].astype(np.float32).values

umap_npy = os.path.join(OUT_DIR, "umap.npy")
if os.path.exists(umap_npy):
    U = np.load(umap_npy)
else:
    if umap is None:
        raise SystemExit("umap-learn not installed and no umap.npy found.")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, metric="correlation",
                        n_components=2, random_state=0)
    U = reducer.fit_transform(X)
    np.save(umap_npy, U)

meta = pd.read_csv(meta_path, index_col=0).reindex(counts.index)
label_col = None
for c in ["Manual_Annotations","Manual_Annotation","ManualLabels",
          "Stirparo_Labels","Stirparo_Label","Label"]:
    if c in meta.columns:
        label_col = c; break
if label_col is None:
    raise SystemExit("Could not find Manual_Annotations in Human_Sample_Info.csv.")
raw_labels = meta[label_col].astype(str).fillna("Unassigned")

CANON = [
    "8-cell","Morula","ICM/TE branch","ICM","Epi/Hyp branch","Hyp",
    "preIm-Epi","Embryonic disc","ExE-Mes","Early TE","Mid TE",
    "Mural TE","Polar TE","cTB","sTB"
]
ALIASES = {
    "8-cell": ["8cell","8-cell","8 cell","eightcell","eight-cell","eight cell","8C","8 c","8cells","8 cells","day3 8c"],
    "Morula": ["morula","late morula","early morula","d4 morula","d4_morula"],
    "ICM/TE branch": ["icm/te branch","icm_te branch","icm-te branch","icm/te","te/icm branch","morula-icm/te branch","branch icm/te","branch1","branch 1"],
    "ICM": ["icm","inner cell mass","d5 icm","day5 icm"],
    "Epi/Hyp branch": ["epi/hyp branch","epi_hyp branch","epi-hyp branch","branch epi/hyp","branch2","branch 2"],
    "Hyp": ["hyp","hypo","hypoblast","primitive endoderm","pre","pe","prE","pre/pe","pr-e","pr_e"],
    "preIm-Epi": ["preim-epi","preim epi","preim_epi","pre-implantation epiblast","preimplantation epiblast","preim epiblast"],
    "Embryonic disc": ["embryonic disc","embryonic-disc","embryonic_disc","postim-epi","postim epi","post-implantation epiblast","postimplantation epiblast","postim_epiblast","epiblast (disc)","embryonic disk"],
    "ExE-Mes": ["exe-mes","exe mes","exe_mes","extra-embryonic mesenchyme","exE-mech","exE-mes"],
    "Early TE": ["early te","te-early","te_early"],
    "Mid TE": ["mid te","te-mid","te_mid"],
    "Mural TE": ["mural te","te mural","te_mural"],
    "Polar TE": ["polar te","te polar","te_polar"],
    "cTB": ["ctb","cytotrophoblast","cyto-trophoblast"],
    "sTB": ["stb","syncytiotrophoblast","syncytio-trophoblast"],
}
def normalize_label(s: str) -> str:
    s0 = s.strip()
    s1 = re.sub(r"[\s\-\_]+", " ", s0).lower()
    for k in CANON:
        if s1 == k.lower():
            return k
    for k, vals in ALIASES.items():
        if s1 in [v.lower() for v in vals]:
            return k
    if "8" in s1 and "cell" in s1: return "8-cell"
    if "hypo" in s1 or "hypoblast" in s1 or re.search(r"\bpe\b|\bpr[e\-_/]?\b", s1): return "Hyp"
    if "icm" in s1 and "te" in s1 and "branch" in s1: return "ICM/TE branch"
    if "epi" in s1 and "hyp" in s1 and "branch" in s1: return "Epi/Hyp branch"
    if "pre" in s1 and "epi" in s1 and ("implant" in s1 or "preim" in s1): return "preIm-Epi"
    if "post" in s1 and "epi" in s1 and ("implant" in s1 or "postim" in s1 or "disc" in s1 or "disk" in s1): return "Embryonic disc"
    if "exe" in s1 and ("mes" in s1 or "mech" in s1): return "ExE-Mes"
    if "mural" in s1 and "te" in s1: return "Mural TE"
    if "polar" in s1 and "te" in s1: return "Polar TE"
    if "ctb" in s1 or "cytotroph" in s1: return "cTB"
    if "syncyt" in s1 or s1 == "stb": return "sTB"
    return s0

labels = raw_labels.map(normalize_label)

palette = {
    "8-cell":"#1f77b4","Morula":"#e377c2","ICM/TE branch":"#2ca02c","ICM":"#d62728",
    "Epi/Hyp branch":"#9467bd","Hyp":"#1f77b4","preIm-Epi":"#bcbd22","Embryonic disc":"#4b0082",
    "ExE-Mes":"#ff7f0e","Early TE":"#17becf","Mid TE":"#ff00aa","Mural TE":"#ff7f0e",
    "Polar TE":"#8c564b","cTB":"#c49c94","sTB":"#aec7e8"
}
order = CANON[:]

plt.figure(figsize=(7,6), dpi=300)
for lab in order:
    m = (labels.values == lab)
    if m.any():
        plt.scatter(U[m,0], U[m,1], s=6, alpha=0.85,
                    label=lab, c=palette.get(lab, "#888888"), linewidths=0)
others = [l for l in pd.unique(labels) if l not in order]
for lab in others:
    m = (labels.values == lab)
    plt.scatter(U[m,0], U[m,1], s=6, alpha=0.6,
                label=lab, c="#bbbbbb", linewidths=0)

plt.xticks([]); plt.yticks([])
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.title("Human embryo (cESFW) — Manual_Annotations (normalized)", pad=10)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
           frameon=False, markerscale=2, fontsize=7)
plt.tight_layout()
out_png = os.path.join(plots_dir, "UMAP_vbranch_manual_annotations.png")
plt.savefig(out_png, bbox_inches="tight")
print("Saved:", out_png)
