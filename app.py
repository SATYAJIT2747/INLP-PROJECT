import streamlit as st
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import numpy as np
import pandas as pd
import copy
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceEncoder(nn.Module):
    def __init__(self, model_name, max_len=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.max_len = max_len

    @staticmethod
    def mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode_texts(self, texts, device):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.transformer(**enc)
        emb = self.mean_pool(out.last_hidden_state, enc['attention_mask'])
        return F.normalize(emb, dim=-1)

    def forward(self, stories, meanings, device):
        hs = self.encode_texts(stories, device)
        hm = self.encode_texts(meanings, device)
        return hs, hm

class OrdinalHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.raw_thresholds = nn.Parameter(torch.linspace(-2.0, 2.0, 4))

    def forward(self, h):
        thresholds = torch.cumsum(F.softplus(self.raw_thresholds), dim=0)
        logit = self.linear(h)
        cum_probs = torch.sigmoid(thresholds - logit)
        zeros = torch.zeros(cum_probs.shape[0], 1, device=h.device)
        ones  = torch.ones(cum_probs.shape[0],  1, device=h.device)
        cum_full = torch.cat([zeros, cum_probs, ones], dim=1)
        probs = cum_full[:, 1:] - cum_full[:, :-1]
        probs = probs.clamp(min=1e-9)
        scores = torch.arange(1, 6, dtype=h.dtype, device=h.device)
        exp    = (probs * scores).sum(dim=-1)
        return probs, exp

class WSDModel(nn.Module):
    def __init__(self, input_dim, h1=512, h2=128, drop=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(drop),
        )
        self.head = OrdinalHead(h2)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

@st.cache_resource(show_spinner=False)
def load_models():
    # Load base encoders once
    base_encoder = SentenceEncoder('sentence-transformers/all-mpnet-base-v2', max_len=128).to('cpu')
    cross_enc = CrossEncoder('cross-encoder/nli-deberta-v3-base', max_length=256)
    
    # Load folds from PT file (weights_only=False because it includes scikit-learn PCA objects)
    models_to_save = torch.load('trial1_models.pt', map_location='cpu', weights_only=False)
    folds_data = []
    
    for fold in range(1, 6):
        data = models_to_save[fold]
        enc = copy.deepcopy(base_encoder)
        enc.transformer.load_state_dict(data['enc_state'])
        enc.to(DEVICE)
        enc.eval()
        
        mlp = WSDModel(202, 512, 128, 0.3).to(DEVICE)
        mlp.load_state_dict(data['mlp_state'])
        mlp.eval()
        
        folds_data.append({
            'enc': enc,
            'pca_l1': data['pca_l1'],
            'pca_had': data['pca_had'],
            'hom_means': data['hom_means'],
            'mlp': mlp
        })
        
    return cross_enc, folds_data

def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=1).astype(np.float32)

def token_overlap_ratio(story_text, meaning_text):
    story_tokens = set(story_text.lower().split())
    meaning_tokens = set(meaning_text.lower().split())
    if len(story_tokens | meaning_tokens) == 0:
        return 0.0
    return len(story_tokens & meaning_tokens) / len(story_tokens | meaning_tokens)

def compute_single_sample(homonym, precontext, sentence, ending, judged_meaning, example_sentence):
    story_text = precontext.strip() + ' ' + sentence.strip()
    if ending:
        story_text += ' ' + ending.strip()
        
    ending_text = ending.strip() if ending else sentence.strip()
    ambiguous_text = sentence.strip()
    meaning_text = judged_meaning.strip() + '. For example: ' + example_sentence.strip()
    
    tok_ratio = token_overlap_ratio(story_text, meaning_text)
    len_rat = len(story_text.split()) / max(len(meaning_text.split()), 1)
    
    cross_enc, folds_data = load_models()
    
    ce_score = cross_enc.predict([[story_text, judged_meaning]], apply_softmax=True)[0, 1]
    
    preds = []
    
    for fold_data in folds_data:
        enc = fold_data['enc']
        mlp = fold_data['mlp']
        
        with torch.no_grad():
            hs = enc.encode_texts([story_text], DEVICE).cpu().numpy()
            hm = enc.encode_texts([meaning_text], DEVICE).cpu().numpy()
            he = enc.encode_texts([ending_text], DEVICE).cpu().numpy()
            ha = enc.encode_texts([ambiguous_text], DEVICE).cpu().numpy()
        
        cos = cosine_sim(hs, hm)
        cos_end = cosine_sim(he, hm)
        cos_amb = cosine_sim(ha, hm)
        
        dlt = np.zeros(1, dtype=np.float32)
        ws_rank = np.zeros(1, dtype=np.float32)
        ce_dlt = np.zeros(1, dtype=np.float32)
        ce_arr = np.array([ce_score], dtype=np.float32)
        
        l1 = np.abs(hs - hm)
        had = hs * hm
        
        l1_p = fold_data['pca_l1'].transform(l1).astype(np.float32)
        had_p = fold_data['pca_had'].transform(had).astype(np.float32)
        
        hom_mean_val = fold_data['hom_means'].get(homonym, 3.0)
        hom_means_arr = np.array([hom_mean_val], dtype=np.float32)
        
        scalars = np.stack([cos, dlt, ce_arr, ws_rank, ce_dlt, cos_end, cos_amb, 
                            np.array([tok_ratio], dtype=np.float32), 
                            np.array([len_rat], dtype=np.float32), 
                            hom_means_arr], axis=1)
        
        X = np.concatenate([scalars, l1_p, had_p], axis=1).astype(np.float32)
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            _, p_exp = mlp(X_t)
            
        preds.append(p_exp.item())
        
    final_score = np.clip(np.mean(preds), 1.0, 5.0)
    return final_score

# --- UI Setup ---
st.set_page_config(page_title="AmbiStory Scorer", layout="centered", initial_sidebar_state="expanded")

with st.sidebar:
    st.subheader("System Status")
    if torch.cuda.is_available():
        st.success(f"GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("CPU Active (No GPU detected)")
    st.info(f"Torch version: {torch.__version__}")

st.markdown("""
    <style>
    .big-score {
        font-size: 80px;
        font-weight: bold;
        text-align: center;
    }
    .score-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    [data-theme="dark"] .score-container {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📖 AmbiStory Plausibility Scorer")
st.markdown("Evaluate the plausibility of a word sense based on the contextual narrative.")

with st.spinner("Initializing models (may take a moment on first run)..."):
    load_models()

tab1, tab2, tab3 = st.tabs(["Single Scenario", "Compare Two Meanings", "Batch Upload (JSON)"])

with tab1:
    t1_file = st.file_uploader("Upload JSON back-end to autofill (optional)", type=["json"], key="t1_file")
    t1_smp = {}
    t1_kstr = "default"
    if t1_file:
        t1_data = json.load(t1_file)
        t1_key = st.selectbox("Select Scenario ID", list(t1_data.keys()), key="t1_sel")
        t1_smp = t1_data.get(t1_key, {})
        t1_kstr = str(t1_key)
        st.info(f"Loaded scenario {t1_key}", icon="📝")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Narrative Setup")
        target_homonym = st.text_input("Target Homonym", value=t1_smp.get("homonym", "drive"), placeholder="e.g., drive", key=f"t1_h_{t1_kstr}")
        precontext = st.text_area("Precontext", value=t1_smp.get("precontext", "Lisa had always been competitive. Every weekend, she dedicated herself to her passion. She believed that her relentless practice would pay off someday."), height=100, key=f"t1_p_{t1_kstr}")
        ambiguous_sentence = st.text_area("Ambiguous Sentence", value=t1_smp.get("sentence", "Her drive was what ultimately got her into the top university."), height=68, key=f"t1_a_{t1_kstr}")
        ending = st.text_area("Ending / Confirmation", value=t1_smp.get("ending", "She made that long trip to show the course coordinators her dedication to going to that university, and they said that was one of the reasons why they accepted her."), height=100, key=f"t1_e_{t1_kstr}")

    with col2:
        st.subheader("Candidate Meaning")
        judged_meaning = st.text_area("Meaning to Evaluate", value=t1_smp.get("judged_meaning", "hitting a golf ball off a tee"), height=68, key=f"t1_jm_{t1_kstr}")
        example_sentence = st.text_area("Example Usage", value=t1_smp.get("example_sentence", "He hit a massive drive on the 18th hole."), height=68, key=f"t1_ex_{t1_kstr}")
        
        st.markdown("---")
        analyze_btn = st.button("🚀 Analyze Plausibility", use_container_width=True, type="primary")

    if analyze_btn:
        if not (target_homonym and ambiguous_sentence and judged_meaning):
            st.error("Please provide at least a Target Homonym, an Ambiguous Sentence, and a Judged Meaning.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting text features...")
            progress_bar.progress(20)
            time.sleep(0.5)  # Little delay for UI effect
            
            status_text.text("Running Transformer inference (MPNet + DeBERTa)...")
            progress_bar.progress(50)
            
            try:
                score = compute_single_sample(
                    target_homonym, precontext, ambiguous_sentence, 
                    ending, judged_meaning, example_sentence
                )
                progress_bar.progress(100)
                status_text.text("Analysis Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Score Animation
                score_placeholder = st.empty()
                st.toast("Model computation finished. Assembling score...", icon="✅")
                
                # Animate the counter from 0 to the final score
                steps = 20
                for i in range(steps + 1):
                    current_val = (score / steps) * i
                    color = "green" if current_val >= 3.5 else "orange" if current_val >= 2.0 else "red"
                    score_placeholder.markdown(f"""
                        <div class="score-container">
                            <div style="text-align: center; font-size: 20px; color: gray;">Estimated Plausibility Score</div>
                            <div class="big-score" style="color: {color};">{current_val:.2f} <span style="font-size: 30px; font-weight: normal; color: gray;">/ 5.0</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.05)

                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab2:
    st.header("Compare Two Candidate Meanings")
    st.markdown("Test two competing meanings against the exact same context to see which scenario is more plausible.")
    
    t2_file = st.file_uploader("Upload JSON back-end to autofill (optional)", type=["json"], key="t2_file")
    t2_smp_a, t2_smp_b = {}, {}
    t2_kast, t2_kbst = "defA", "defB"
    if t2_file:
        t2_data = json.load(t2_file)
        sc1, sc2 = st.columns(2)
        k_a = sc1.selectbox("Select Scenario ID for Meaning A", list(t2_data.keys()), key="t2_sa")
        k_b = sc2.selectbox("Select Scenario ID for Meaning B", list(t2_data.keys()), key="t2_sb")
        t2_smp_a, t2_smp_b = t2_data.get(k_a, {}), t2_data.get(k_b, {})
        t2_kast, t2_kbst = str(k_a), str(k_b)
        st.info("Loaded fields from chosen IDs. Context will be inherited from Meaning A.", icon="📝")

    comp_col1, comp_col2, comp_col3 = st.columns([1.2, 1, 1])

    with comp_col1:
        st.subheader("Shared Context")
        c_target_homonym = st.text_input("Target Homonym", value=t2_smp_a.get("homonym", "match"), key=f"c_hom_{t2_kast}")
        c_precontext = st.text_area("Precontext", value=t2_smp_a.get("precontext", "The rain finally stopped, and the crowd began to gather. John checked the pockets of his coat nervously."), height=100, key=f"c_pre_{t2_kast}")
        c_ambiguous_sentence = st.text_area("Ambiguous Sentence", value=t2_smp_a.get("sentence", "The match was striking."), height=68, key=f"c_amb_{t2_kast}")
        c_ending = st.text_area("Ending", value=t2_smp_a.get("ending", "But the box was wet, and it wouldn't light."), height=100, key=f"c_end_{t2_kast}")

    with comp_col2:
        st.subheader("Meaning A")
        c_meaning_a = st.text_area("Meaning to Evaluate", value=t2_smp_a.get("judged_meaning", "lighter consisting of a thin piece of wood tipped with combustible chemical"), height=68, key=f"c_mean_a_{t2_kast}")
        c_example_a = st.text_area("Example Usage", value=t2_smp_a.get("example_sentence", "He struck a match to light the candle."), height=68, key=f"c_ex_a_{t2_kast}")

    with comp_col3:
        st.subheader("Meaning B")
        c_meaning_b = st.text_area("Meaning to Evaluate", value=t2_smp_b.get("judged_meaning", "a formal contest in which two or more persons or teams compete"), height=68, key=f"c_mean_b_{t2_kbst}")
        c_example_b = st.text_area("Example Usage", value=t2_smp_b.get("example_sentence", "The teams prepared for the upcoming match."), height=68, key=f"c_ex_b_{t2_kbst}")
        
    st.markdown("---")
    compare_btn = st.button("⚖️ Compare Candidates", use_container_width=True, type="primary", key="c_btn")

    if compare_btn:
        if not (c_target_homonym and c_ambiguous_sentence and c_meaning_a and c_meaning_b):
            st.error("Please fill in the Context, Meaning A, and Meaning B.")
        else:
            progress_bar_c = st.progress(0)
            status_text_c = st.empty()
            
            try:
                status_text_c.text("Evaluating Meaning A...")
                score_a = compute_single_sample(
                    c_target_homonym, c_precontext, c_ambiguous_sentence, 
                    c_ending, c_meaning_a, c_example_a
                )
                progress_bar_c.progress(50)
                
                status_text_c.text("Evaluating Meaning B...")
                score_b = compute_single_sample(
                    c_target_homonym, c_precontext, c_ambiguous_sentence, 
                    c_ending, c_meaning_b, c_example_b
                )
                progress_bar_c.progress(100)
                status_text_c.text("Comparison Complete!")
                time.sleep(0.5)
                progress_bar_c.empty()
                status_text_c.empty()
                
                rc1, rc2 = st.columns(2)
                
                def render_score(score, label):
                    color = "green" if score >= 3.5 else "orange" if score >= 2.0 else "red"
                    return f"""
                        <div class="score-container" style="border: {"3px solid #4CAF50" if score >= 3.5 else "none"}">
                            <div style="text-align: center; font-size: 20px; color: gray;">{label}</div>
                            <div class="big-score" style="color: {color};">{score:.2f} <span style="font-size: 24px; color: gray;">/ 5.0</span></div>
                        </div>
                    """
                
                with rc1:
                    st.markdown(render_score(score_a, "Meaning A Score"), unsafe_allow_html=True)
                with rc2:
                    st.markdown(render_score(score_b, "Meaning B Score"), unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                if abs(score_a - score_b) < 0.1:
                    st.info("Both meanings are almost equally plausible!", icon="⚖️")
                elif score_a > score_b:
                    st.success("**Meaning A** is more likely for this scenario!", icon="🏆")
                else:
                    st.success("**Meaning B** is more likely for this scenario!", icon="🏆")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab3:
    st.header("Batch Scenarios via JSON")
    st.markdown("Upload an AmbiStory JSON dictionary where keys are sample IDs and values contain `homonym`, `precontext`, `sentence`, `ending`, `judged_meaning`, and `example_sentence`.")
    
    uploaded_file = st.file_uploader("Upload AmbiStory JSON file", type=["json"])
    
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        st.success(f"Successfully loaded {len(data)} scenarios!")
        
        if st.button("Evaluate All Scenarios", type="primary"):
            progress_text = "Evaluating scenarios. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            results = []
            keys = list(data.keys())
            
            for i, key in enumerate(keys):
                item = data[key]
                try:
                    pred_score = compute_single_sample(
                        item.get("homonym", ""),
                        item.get("precontext", ""),
                        item.get("sentence", ""),
                        item.get("ending", ""),
                        item.get("judged_meaning", ""),
                        item.get("example_sentence", "")
                    )
                    
                    actual_score = item.get("average", None)
                    
                    results.append({
                        "ID": key,
                        "Homonym": item.get("homonym", ""),
                        "Sentence": item.get("sentence", ""),
                        "Predicted Score": round(pred_score, 2),
                        "Actual Score": round(actual_score, 2) if actual_score is not None else "N/A"
                    })
                except Exception as e:
                    st.error(f"Error evaluating sample {key}: {e}")
                    
                my_bar.progress((i + 1) / len(keys), text=f"Processed {i + 1}/{len(keys)} scenarios...")
            
            my_bar.empty()
            
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Optional: Calculate metrics if ground truth was provided for all
            if "Actual Score" in df_results.columns and not df_results["Actual Score"].eq("N/A").any():
                from scipy.stats import spearmanr
                truths = df_results["Actual Score"].values
                preds = df_results["Predicted Score"].values
                corr, _ = spearmanr(truths, preds)
                st.metric("Spearman Correlation (Batch)", f"{corr:.4f}")

