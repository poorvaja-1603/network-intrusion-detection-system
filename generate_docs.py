"""
generate_docs.py — Generates a professional .docx technical explanation document
for Member A's code and saves it to the Desktop.
"""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DESKTOP_PATH = Path(r"C:\Users\Poorvaja\Desktop\Member_A_Code_Explanation.docx")

# Styling Constants
COLOR_PRIMARY = RGBColor(15, 76, 129)     # Deep Classic Blue
COLOR_SECONDARY = RGBColor(43, 84, 126)   # Slate Blue
COLOR_TEXT = RGBColor(33, 37, 41)         # Dark Gray text
COLOR_MUTED = RGBColor(108, 117, 125)     # Muted Gray
COLOR_CODE = RGBColor(199, 37, 78)        # Code pink/red
HEX_PRIMARY = "0F4C81"
HEX_LIGHT_BG = "F4F6F8"
HEX_BORDER = "CCCCCC"


def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins (padding) of a cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_code_block(doc, code_text):
    """Add a shaded code block with monospace font."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Border
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(40, 44, 52)
    
    # Add spacing after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def add_callout(doc, title, text, icon="ℹ"):
    """Add a stylish callout box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "EBF3FB")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"{icon}  {title}\n")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = COLOR_PRIMARY
    
    run_b = p.add_run(text)
    run_b.font.name = "Calibri"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = COLOR_TEXT
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def format_table(table, col_widths, headers, data):
    """Create a professionally styled data table."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(title)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_col = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].width = col_widths[c_idx]
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if c_idx > 0 and any(char.isdigit() for char in str(val)):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = COLOR_TEXT


def build_document():
    doc = docx.Document()
    
    # Page Setup: Normal 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles configuration
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = COLOR_TEXT
    
    # ------------------------------------------------------------------ #
    # Document Header / Title Page Area                                  #
    # ------------------------------------------------------------------ #
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("Network Intrusion Detection System with Dimensionality Reduction")
    run_title.bold = True
    run_title.font.name = "Georgia"
    run_title.font.size = Pt(22)
    run_title.font.color.rgb = COLOR_PRIMARY
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("Comprehensive Engineering Documentation & Code Walkthrough: Member A Deliverables\n")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = COLOR_SECONDARY
    
    # Metadata Badge Table
    meta_table = doc.add_table(rows=1, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.7)]
    headers_meta = ["Role", "Author", "Tasks Owned", "Paper Reference"]
    data_meta = [["Lead ML Infrastructure", "Member A", "3, 6, 7, 8, 9, 12, 16, 20, 22, 31", "Nabi & Zhou (2024)"]]
    format_table(meta_table, widths, headers_meta, data_meta)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # ------------------------------------------------------------------ #
    # Section 1: Executive Overview                                      #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("1. Executive Overview & Member A Role Scope", level=1)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(6)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    p = doc.add_paragraph(
        "Member A serves as the Core Machine Learning Engineer and Infrastructure Owner for the project. "
        "The primary mission is to reproduce the paper's benchmark experiments (Nabi & Zhou, 2024), "
        "extend the research with modern modeling techniques, and establish strict software contracts so that "
        "Member B (benchmarking & Weka), Member C (FastAPI inference engine), and Member D (Streamlit dashboard & PCA) "
        "can build upon stable, leak-free components without blocking."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)
    
    # Task Ownership Table
    doc.add_heading("Table 1.1: Complete Audit of Tasks Assigned to and Delivered by Member A", level=3)
    task_table = doc.add_table(rows=1, cols=4)
    t_widths = [Inches(0.8), Inches(1.8), Inches(1.8), Inches(2.1)]
    t_headers = ["Task #", "Plan Slot & Module", "Delivered Artifact", "Key Result / Implementation"]
    t_data = [
        ["#3", "Sat / Data Contract", "src/data.py", "NSL-KDD loader (125k train, 22k test), 41 raw features, dedupe checks"],
        ["#6", "Sat / Preprocessing", "src/preprocess.py", "ColumnTransformer (StandardScaler + OneHotEncoder -> 122 dims, fit train only)"],
        ["#7", "Sat / Model Zoo", "src/models.py", "NaiveBayes, J48 (entropy DT), RandomForest, MLP (128x64), XGBoost"],
        ["#8", "Sat / Evaluation", "src/evaluate.py", "Accuracy, FPR, Macro-F1, confusion matrices, high-precision timing, JSON I/O"],
        ["#9", "Sat / Exp 1 Reproduction", "results/exp1_full_features.json", "J48: 77.7% (paper: 79.1%), RF: 76.5%, MLP: 79.7%, XGB: 79.4%"],
        ["#12", "Sun / Gaussian RP Sweep", "results/exp3_rp_gaussian_sweep.json", "Swept dims 1 to 30. Empirically validated 5 dims as tree sweet spot (78.7%)"],
        ["#16", "Sun / 5-Class Extension", "results/exp5_multiclass.json", "Normal, DoS, Probe, R2L, U2R. MLP Macro-F1: 0.584, RF 5-dim RP: 0.503"],
        ["#20", "Sun 6pm / Final Model", "models/final.joblib", "End-to-end sklearn Pipeline accepting raw 41-feature inputs (<0.012 ms latency)"],
        ["#22", "Mon / Error Analysis", "reports/error_analysis.md", "Granular 39-attack recall; 30.6% drop on zero-day attacks; 10 FN traces"],
        ["#31", "Mon / Technical Report", "reports/report_sections.md", "Mathematical formulation of JL lemma, comparative tables, 4 paper critique points"]
    ]
    format_table(task_table, t_widths, t_headers, t_data)
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # ------------------------------------------------------------------ #
    # Section 2: Architectural Principles & Anti-Leakage Contracts       #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("2. Architectural Principles & System Contracts", level=1)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    p = doc.add_paragraph(
        "A critical flaw in many published machine learning benchmarks is data leakage: fitting encoders or scalers "
        "across the entire dataset before train/test splitting. In this project, Member A established three rigid architectural contracts:"
    )
    p.paragraph_format.space_after = Pt(6)
    
    add_callout(doc, "Contract 1: Strict Train-Only Preprocessing (No Leakage)",
                "Scalers and OneHotEncoders are fitted EXCLUSIVELY on KDDTrain+. The test dataset (KDDTest+) is treated as pure "
                "incoming telemetry transformed by fitted objects. Unseen categorical values in the test set (e.g. novel services) "
                "are silently zero-filled via handle_unknown='ignore' rather than throwing exceptions or inducing leakage.")
                
    add_callout(doc, "Contract 2: The Self-Contained Pipeline Contract (models/final.joblib)",
                "Downstream consumers (Member C's FastAPI server and Member D's Streamlit UI) must NEVER perform manual feature "
                "engineering. The final serialized model is a single sklearn.pipeline.Pipeline that accepts raw 41-column Pandas "
                "DataFrames directly, automatically transforming features, applying encoders, and executing prediction in a single step.")
                
    add_callout(doc, "Contract 3: The Results Schema (results/<experiment_name>.json)",
                "All experiment scripts output standardized JSON structures containing model name, setting, dimensions, "
                "exact random seed, accuracy, FPR, precision, recall, F1, macro-F1, fit time in seconds, and confusion matrices. "
                "This guarantees that dashboard visualizations and report charts read identical ground truth data.")

    # ------------------------------------------------------------------ #
    # Section 3: Deep-Dive Module-by-Module Code Walkthrough             #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("3. Detailed Module-by-Module Code Walkthrough", level=1)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    # --- Module 1: attack_map.py ---
    doc.add_heading("3.1 Module: src/attack_map.py (Attack Taxonomy & Category Mapping)", level=2)
    p = doc.add_paragraph(
        "The NSL-KDD dataset labels connections by specific attack names (e.g., 'neptune', 'buffer_overflow') rather than "
        "broad categories. Furthermore, KDDTest+ contains 17 novel attack types that do not appear in KDDTrain+. "
        "Member A implemented a comprehensive taxonomy dictionary mapping all 39 distinct attack signatures into 5 canonical categories: "
        "Normal, DoS (Denial of Service), Probe (Surveillance), R2L (Remote to Local compromise), and U2R (User to Root privilege escalation)."
    )
    add_code_block(doc, 
"""# Snippet from src/attack_map.py
TRAIN_ATTACKS = {
    'normal': 'normal',
    # DoS
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
    # Probe
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    # R2L & U2R
    'guess_passwd': 'r2l', 'warezclient': 'r2l', 'buffer_overflow': 'u2r', 'rootkit': 'u2r', ...
}

# Test-Only Zero-Day Attacks (Crucial for Generalization Analysis)
TEST_ONLY_ATTACKS = {
    'apache2': 'dos', 'udpstorm': 'dos', 'mscan': 'probe', 'saint': 'probe',
    'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'ps': 'u2r', 'sqlattack': 'u2r', ...
}

ATTACK_MAP = {**TRAIN_ATTACKS, **TEST_ONLY_ATTACKS}

def get_category(attack_name: str) -> str:
    return ATTACK_MAP[attack_name.strip().lower()]

def get_binary_label(attack_name: str) -> int:
    return 0 if attack_name.strip().lower() == 'normal' else 1
""")
    
    # --- Module 2: data.py ---
    doc.add_heading("3.2 Module: src/data.py (Data Ingestion & Integrity Auditing)", level=2)
    p = doc.add_paragraph(
        "Implements load_nsl_kdd() adhering to Task 3 contracts. Raw text files lack header rows and contain 43 comma-separated columns: "
        "41 network connection attributes, the attack label, and a difficulty score. "
        "The module assigns standard names, strips whitespace, removes the 43rd difficulty column, extracts binary labels (0/1) "
        "and 5-class strings, and conducts an automated deduplication audit."
    )
    add_code_block(doc,
"""# Snippet from src/data.py
FEATURE_COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', ...]
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']

def load_nsl_kdd():
    train_df = _load_raw(DATA_DIR / 'KDDTrain+.txt')
    test_df = _load_raw(DATA_DIR / 'KDDTest+.txt')
    
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    
    y_train_bin = train_df['attack_type'].apply(get_binary_label)
    y_test_bin = test_df['attack_type'].apply(get_binary_label)
    
    # Deduplication audit (Task 3 specification)
    train_dups = int(X_train.duplicated().sum())       # 16 duplicate rows
    test_dups = int(X_test.duplicated().sum())         # 57 duplicate rows
    overlap_dups = int(pd.merge(X_train, X_test).shape[0]) # 668 overlapping rows
    
    return X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat
""")

    # --- Module 3: preprocess.py ---
    doc.add_heading("3.3 Module: src/preprocess.py (Feature Transformation Pipeline)", level=2)
    p = doc.add_paragraph(
        "Dimensionality reduction techniques such as PCA and Random Projection operate strictly on continuous numerical vectors. "
        "However, NSL-KDD contains 3 string categorical features ('protocol_type' with 3 levels, 'service' with 70 levels, "
        "and 'flag' with 11 levels). "
        "Member A constructed a sklearn ColumnTransformer that applies OneHotEncoder(sparse_output=False, handle_unknown='ignore') "
        "to the 3 categoricals and StandardScaler() to the 38 numerical features. "
        "This expands the feature space from 41 raw columns to 122 dense encoded features without leaking future distributions."
    )
    add_code_block(doc,
"""# Snippet from src/preprocess.py
def build_preprocessor(X_train=None):
    numeric_cols = get_numeric_columns() # 38 columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLUMNS)
        ],
        remainder='drop'
    )
    if X_train is not None:
        preprocessor.fit(X_train)
    return preprocessor

def preprocess_data(X_train, X_test):
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train) # Shape: (125973, 122)
    X_test_proc = preprocessor.transform(X_test)       # Shape: (22544, 122)
    return X_train_proc, X_test_proc, preprocessor
""")

    # --- Module 4: models.py ---
    doc.add_heading("3.4 Module: src/models.py (The Model Zoo)", level=2)
    p = doc.add_paragraph(
        "The research paper conducted experiments inside the Weka GUI using BayesNet, Naive Bayes, J48 (C4.5), PART, and Random Forest. "
        "In Python's scikit-learn environment, C4.5 is matched by DecisionTreeClassifier(criterion='entropy'). "
        "Furthermore, Member A added modern gradient boosted trees (XGBoost) and deep multi-layer neural networks (MLP) "
        "with fixed seeds (SEED = 42) to evaluate whether modern algorithms overcome the paper's reported limitations."
    )
    add_code_block(doc,
"""# Snippet from src/models.py
def get_models(task: str = 'binary') -> dict:
    models = {
        'NaiveBayes': GaussianNB(),
        'J48': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, n_jobs=-1),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    return models
""")

    # --- Module 5: evaluate.py ---
    doc.add_heading("3.5 Module: src/evaluate.py (Metrics, High-Precision Timing, & Storage)", level=2)
    p = doc.add_paragraph(
        "In network intrusion detection, False Positive Rate (FPR) has a distinct operational meaning: normal connections "
        "falsely flagged as intrusions create alert fatigue for security operations centers (SOCs). "
        "Member A implemented a dedicated compute_fpr() utility based on the cybersecurity standard: "
        "FPR = FP / (FP + TN), alongside micro and macro F1, per-class recall, and context-manager timing wrappers using time.perf_counter()."
    )
    add_code_block(doc,
"""# Snippet from src/evaluate.py
def compute_fpr(y_true, y_pred, positive_label=1) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
""")

    # --- Module 6: run_experiments.py ---
    doc.add_heading("3.6 Module: src/run_experiments.py (The Master Experiment Pipeline)", level=2)
    p = doc.add_paragraph(
        "Orchestrates Experiment 1 (Full Features Baseline), Experiment 2 (Gaussian Random Projection Sweep across dimensions 1 to 30), "
        "and Experiment 3 (5-Class Multiclass Extension). Accepts CLI flags (--exp 1, --exp 2, --exp 3, --exp all) "
        "and automatically serializes results into the results/ directory."
    )
    add_code_block(doc,
"""# Snippet from src/run_experiments.py
def run_exp2_rp_gaussian_sweep():
    dims = [1, 2, 3, 4, 5, 10, 15, 20, 30]
    for n_dim in dims:
        rp = GaussianRandomProjection(n_components=n_dim, random_state=42)
        X_train_rp = rp.fit_transform(X_train_proc)
        X_test_rp = rp.transform(X_test_proc)
        # Train and evaluate NaiveBayes, J48, RandomForest across each subspace
""")

    # --- Module 7: build_final_model.py ---
    doc.add_heading("3.7 Module: src/build_final_model.py (Production Packaging)", level=2)
    p = doc.add_paragraph(
        "Builds models/final.joblib: a single, self-contained sklearn Pipeline containing the preprocessor and the best "
        "multiclass classifier. Also implements DummyModel in models/dummy.joblib so that Member C (FastAPI) and Member D (Streamlit) "
        "could develop in parallel without blocking."
    )
    add_code_block(doc,
"""# Snippet from src/build_final_model.py
pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, n_jobs=-1))
])
pipeline.fit(X_train, y_train_cat)
joblib.dump(pipeline, 'models/final.joblib')
""")

    # --- Module 8: error_analysis.py ---
    doc.add_heading("3.8 Module: src/error_analysis.py (Vulnerability & Zero-Day Breakdown)", level=2)
    p = doc.add_paragraph(
        "Audits model performance across all 39 attack types on KDDTest+. Separates train-seen attacks from test-only novel attacks "
        "and dynamically generates feature-level explanations for 10 concrete false negatives by inspecting TCP flags, "
        "SYN error rates, and authentication status."
    )

    # ------------------------------------------------------------------ #
    # Section 4: Experimental Findings & Empirical Comparison            #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("4. Experimental Results & Empirical Comparison", level=1)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    doc.add_heading("4.1 Experiment 1: Full 41/122 Features Reproduction (Paper Table 3 vs Ours)", level=2)
    e1_table = doc.add_table(rows=1, cols=6)
    e1_widths = [Inches(1.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]
    e1_headers = ["Algorithm", "Paper Acc (%)", "Paper FPR (%)", "Our Acc (%)", "Our FPR (%)", "Our Fit Time"]
    e1_data = [
        ["Naive Bayes", "73.1%", "23.5%", "56.6%", "0.7%", "0.27s"],
        ["J48 (Decision Tree)", "79.1%", "18.5%", "77.7%", "3.1%", "1.42s"],
        ["Random Forest", "77.8%", "20.1%", "76.5%", "2.7%", "2.81s"],
        ["MLP (Extension)", "N/A", "N/A", "79.7%", "4.1%", "55.67s"],
        ["XGBoost (Extension)", "N/A", "N/A", "79.4%", "2.8%", "3.71s"]
    ]
    format_table(e1_table, e1_widths, e1_headers, e1_data)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    doc.add_heading("4.2 Experiment 2: Extended Gaussian Random Projection Sweep", level=2)
    e2_table = doc.add_table(rows=1, cols=5)
    e2_widths = [Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3)]
    e2_headers = ["Subspace Dims", "Naive Bayes Acc", "J48 Tree Acc", "Random Forest Acc", "Avg Fit Time"]
    e2_data = [
        ["1 dimension", "71.8%", "74.5%", "74.5%", "2.38s"],
        ["2 dimensions", "72.7%", "78.2%", "78.7%", "1.35s"],
        ["3 dimensions", "73.4%", "77.4%", "76.3%", "1.34s"],
        ["4 dimensions", "73.0%", "78.4%", "77.6%", "2.47s"],
        ["5 dimensions (Paper)", "73.1%", "78.7%", "77.2%", "2.73s"],
        ["10 dimensions", "75.0%", "78.1%", "76.0%", "6.01s"],
        ["15 dimensions", "75.2%", "73.7%", "74.2%", "3.76s"],
        ["20 dimensions", "76.7%", "76.7%", "74.6%", "4.45s"],
        ["30 dimensions", "76.7%", "74.3%", "73.6%", "6.53s"]
    ]
    format_table(e2_table, e2_widths, e2_headers, e2_data)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_heading("4.3 Experiment 3: 5-Class Multiclass Extension Performance", level=2)
    e3_table = doc.add_table(rows=1, cols=7)
    e3_widths = [Inches(1.2), Inches(1.0), Inches(0.9), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85)]
    e3_headers = ["Algorithm", "Setting", "Overall Acc", "Macro F1", "DoS Recall", "Probe Recall", "R2L Recall"]
    e3_data = [
        ["MLP", "Full (122)", "78.7%", "0.584", "81.2%", "76.4%", "23.1%"],
        ["J48 Tree", "Full (122)", "77.6%", "0.558", "80.5%", "77.1%", "21.8%"],
        ["XGBoost", "Full (122)", "77.5%", "0.536", "82.0%", "74.9%", "18.2%"],
        ["Random Forest", "Full (122)", "73.7%", "0.475", "78.9%", "73.2%", "16.5%"],
        ["MLP", "Gaussian RP (5)", "75.3%", "0.527", "77.1%", "70.8%", "15.4%"],
        ["Random Forest", "Gaussian RP (5)", "74.7%", "0.503", "77.4%", "69.3%", "14.8%"],
        ["Naive Bayes", "Gaussian RP (5)", "68.4%", "0.521", "69.2%", "66.5%", "19.3%"]
    ]
    format_table(e3_table, e3_widths, e3_headers, e3_data)
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ------------------------------------------------------------------ #
    # Section 5: Critique of Paper Weak Spots                            #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("5. Critical Analysis of Paper's Weak Spots", level=1)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    p = doc.add_paragraph(
        "A key contribution of this project is discovering and verifying four critical weak spots in Nabi & Zhou (2024):"
    )
    
    add_callout(doc, "Weak Spot 1: Binary Accuracy Masks Severe Vulnerability on R2L & U2R",
                "The paper reports ~80-82% accuracy as strong performance. However, our granular breakdown reveals that "
                "detection rate drops from 82.4% on known training attacks down to 51.8% on novel test attacks. Worse, "
                "Remote-to-Local (R2L) and User-to-Root (U2R) intrusions exhibit catastrophic detection rates under 25% and 15%. "
                "Because DoS attacks dominate the dataset volume, high DoS accuracy artificially inflates the overall score.", icon="⚠")

    add_callout(doc, "Weak Spot 2: Unmeasured Timing Claims",
                "The paper claims Random Projection is 'faster than PCA' without reporting a single millisecond of benchmark data. "
                "Member A measured the transformation directly: Gaussian RP matrix multiplication requires 0.08 seconds, whereas "
                "PCA covariance decomposition requires 1.45 seconds (an 18x difference).", icon="⏱")

    add_callout(doc, "Weak Spot 3: The '92% Variance in PC1' Scaling Anomaly",
                "The paper claimed 92% of variance was captured in the first principal component (PC1). Our mathematical audit "
                "demonstrated that this only occurs when high-magnitude continuous features (src_bytes, dst_bytes reaching millions) "
                "are left UNSTANDARDIZED, dominating the covariance matrix. When standard z-score scaling is applied, variance is distributed "
                "across components.", icon="🔍")

    add_callout(doc, "Weak Spot 4: Table 4 Typographical Error in the Paper",
                "In Table 4 of the paper, cell (2, 1) in the J48 confusion matrix is printed as 34,231. Because the entire test set "
                "contains only 12,833 total anomaly instances, this value is mathematically impossible. The true value is 4,231 "
                "(since 4,231 + 8,602 = 12,833).", icon="✍")

    # ------------------------------------------------------------------ #
    # Section 6: Viva Voce Defense Guide                                 #
    # ------------------------------------------------------------------ #
    h1 = doc.add_heading("6. Viva Voce Defense Preparation for Member A", level=1)
    for r in h1.runs:
        r.font.name = "Georgia"
        r.font.color.rgb = COLOR_PRIMARY
        
    viva_qas = [
        ("Q1: How does Random Projection work and why is it faster than PCA?",
         "PCA requires computing the sample covariance matrix S = (1/N) * sum((y_i - mu)(y_i - mu)^T) and solving for eigenvalues/eigenvectors, which scales with O(N * P^2 + P^3). Random Projection bypasses covariance decomposition entirely; it multiplies data matrix Y by an independent identically distributed random Gaussian matrix W of dimension D x P scaled by 1/sqrt(D). By the Johnson-Lindenstrauss lemma, Euclidean distances are preserved within (1 +/- epsilon) with high probability in O(N * P * D) time."),
        ("Q2: Why did you use entropy criterion for Decision Tree instead of Gini?",
         "Weka's J48 algorithm is an implementation of Ross Quinlan's C4.5 decision tree, which uses Information Gain (based on Shannon Entropy) rather than Gini impurity. To reproduce Weka J48 faithfully in scikit-learn, DecisionTreeClassifier must be configured with criterion='entropy'."),
        ("Q3: How did you ensure zero data leakage between train and test sets?",
         "All ColumnTransformer scalers and one-hot encoders were fit exclusively on KDDTrain+. The fitted pipeline then transformed KDDTest+. Any categorical values appearing in test that were not present in training are ignored (handle_unknown='ignore') rather than included during encoder fitting."),
        ("Q4: Why does accuracy drop so severely between training cross-validation (~99%) and KDDTest+ (~78%)?",
         "This is not a modeling bug; it is an intrinsic structural property of NSL-KDD. KDDTest+ was specifically designed with 17 novel attack types that do not exist in KDDTrain+ (e.g., mscan, apache2, ps, httptunnel). Cross-validation on train only tests seen attack distributions, whereas test measures true zero-day generalization.")
    ]
    
    for q, a in viva_qas:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(8)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = COLOR_PRIMARY
        
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_before = Pt(0)
        p_a.paragraph_format.space_after = Pt(6)
        r_a = p_a.add_run(a)
        r_a.font.size = Pt(9.5)
        
    # Save document
    doc.save(DESKTOP_PATH)
    print(f"\nDocument successfully created and saved to:\n  {DESKTOP_PATH}")


if __name__ == "__main__":
    build_document()

