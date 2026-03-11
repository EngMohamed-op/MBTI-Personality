import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import os
import random
import nltk

# -- page setup (must come first) --
st.set_page_config(page_title="MBTI Personality Dashboard", page_icon="🧠", layout="wide")

# grab stopwords for the word cloud
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

# ──────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(102,126,234,.35);
}
div[data-testid="stMetric"] label { color: rgba(255,255,255,.85) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; font-weight: 700; }

section[data-testid="stSidebar"] { background: #1a1a2e; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #667eea; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# consistent plotly look
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Segoe UI, sans-serif",
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORS = px.colors.qualitative.Pastel
ACCENT = ["#667eea", "#f093fb"]

DIMENSIONS = ["IE", "NS", "TF", "JP"]


# ──────────────────────────────────────────────
# Data loading & prep
# ──────────────────────────────────────────────
def clean_text(text):
    """Strip URLs, separators, MBTI keywords, and special chars."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\|\|\|", " ", text)
    all_types = (
        "infp|infj|intp|intj|entp|enfp|istp|isfp"
        "|entj|istj|enfj|isfj|estp|esfp|estj|esfj"
    )
    text = re.sub(rf"\b({all_types})\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner="Loading & processing data …")
def load_data():
    # try local file first, then kagglehub as fallback
    csv_path = None
    for p in ["mbti_1.csv", os.path.join(os.path.dirname(__file__), "mbti_1.csv")]:
        if os.path.isfile(p):
            csv_path = p
            break
    if csv_path is None:
        try:
            import kagglehub
            csv_path = os.path.join(
                kagglehub.dataset_download("datasnaek/mbti-type"), "mbti_1.csv"
            )
        except Exception:
            pass
    if not csv_path or not os.path.isfile(csv_path):
        st.error("Could not find **mbti_1.csv**. Place it next to app.py or install kagglehub.")
        st.stop()

    df = pd.read_csv(csv_path)
    df["clean_posts"] = df["posts"].apply(clean_text)

    # feature engineering
    df["Word_count"] = df["clean_posts"].str.split().str.len()
    df["avg_word_length"] = df["clean_posts"].apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    )
    df["lexical_diversity"] = df["clean_posts"].apply(
        lambda t: len(set(t.split())) / max(len(t.split()), 1)
    )
    df["self_reference_ratio"] = df["clean_posts"].apply(
        lambda t: sum(w in {"i", "me", "my", "myself"} for w in t.split()) / max(len(t.split()), 1)
    )
    df["sentiment"] = df["clean_posts"].apply(lambda t: TextBlob(t).sentiment.polarity)

    # split MBTI dimensions
    for i, col in enumerate(DIMENSIONS):
        df[col] = df["type"].str[i]

    df["punctuation_count"] = df["posts"].str.count(r"[!?]")
    df["link_count"] = df["posts"].str.count("http") + df["posts"].str.count("#")

    return df


# ──────────────────────────────────────────────
# ML: train classifiers for each MBTI dimension
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models …")
def train_models(_df):
    """Train one TF-IDF + LogisticRegression per dimension and return them along with raw test sets."""
    models = {}
    vectorizers = {}
    test_sets = {}  # raw text + indices for the demo

    for dim in DIMENSIONS:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vec.fit_transform(_df["clean_posts"])
        y = _df[dim]

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, _df.index, test_size=0.2, random_state=42, stratify=y
        )

        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X_train, y_train)

        models[dim] = clf
        vectorizers[dim] = vec
        test_sets[dim] = idx_test  # keep test indices for sampling

    return models, vectorizers, test_sets


def predict_mbti(text, models, vectorizers):
    """Predict full MBTI type from raw text. Returns (predicted_type, confidence_dict)."""
    cleaned = clean_text(text)
    predicted = ""
    confidence = {}

    for dim in DIMENSIONS:
        vec = vectorizers[dim]
        clf = models[dim]
        x = vec.transform([cleaned])
        proba = clf.predict_proba(x)[0]
        pred_label = clf.classes_[np.argmax(proba)]
        pred_prob = np.max(proba)
        predicted += pred_label
        confidence[dim] = (pred_label, pred_prob)

    return predicted, confidence


# ──────────────────────────────────────────────
# Chart builders
# ──────────────────────────────────────────────
def type_distribution_chart(df):
    counts = df["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.bar(counts, x="type", y="count", color="type",
                 color_discrete_sequence=COLORS,
                 title="Personality Type Distribution",
                 labels={"type": "MBTI Type", "count": "Users"})
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    return fig


def ie_pie(df):
    ie = df["IE"].value_counts().reset_index()
    ie.columns = ["dim", "count"]
    ie["label"] = ie["dim"].map({"I": "Introvert", "E": "Extrovert"})
    fig = px.pie(ie, values="count", names="label",
                 color_discrete_sequence=ACCENT,
                 title="Introvert vs Extrovert", hole=.45)
    fig.update_layout(**CHART_LAYOUT)
    return fig


def wordcount_hist(df):
    fig = px.histogram(df, x="Word_count", nbins=50,
                       color_discrete_sequence=["#667eea"],
                       title="Post Word Count Distribution",
                       labels={"Word_count": "Word Count"})
    fig.update_layout(**CHART_LAYOUT)
    return fig


def wordcount_box(df):
    fig = px.box(df, x="type", y="Word_count", color="type",
                 color_discrete_sequence=COLORS,
                 title="Word Count by Personality Type",
                 labels={"type": "MBTI Type", "Word_count": "Word Count"})
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    return fig


def sentiment_by_type(df):
    avg = df.groupby("type")["sentiment"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="type", y="sentiment", color="sentiment",
                 color_continuous_scale="RdYlGn",
                 title="Average Sentiment by MBTI Type",
                 labels={"type": "MBTI Type", "sentiment": "Avg Sentiment"})
    fig.update_layout(**CHART_LAYOUT)
    return fig


def tf_sentiment_violin(df):
    fig = px.violin(df, x="TF", y="sentiment", color="TF",
                    color_discrete_sequence=ACCENT,
                    title="Sentiment: Thinking vs Feeling",
                    labels={"TF": "", "sentiment": "Polarity"},
                    box=True, points=False)
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    fig.update_xaxes(tickvals=["T", "F"], ticktext=["Thinking (T)", "Feeling (F)"])
    return fig


def correlation_heatmap(df):
    cols = ["Word_count", "avg_word_length", "lexical_diversity",
            "self_reference_ratio", "sentiment", "punctuation_count", "link_count"]
    fig = px.imshow(df[cols].corr(), text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Heatmap")
    fig.update_layout(**CHART_LAYOUT, height=550)
    return fig


def avg_feature_chart(df, feature, title):
    avg = df.groupby("type")[feature].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="type", y=feature, color="type",
                 color_discrete_sequence=COLORS, title=title,
                 labels={"type": "MBTI Type", feature: f"Avg {feature}"})
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    return fig


def make_wordcloud(df):
    text = " ".join(df["clean_posts"])
    wc = WordCloud(width=900, height=450, background_color="#0e1117",
                   colormap="cool", max_words=200, stopwords=STOP_WORDS,
                   contour_width=1, contour_color="#667eea").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    return fig


def dimension_bar(df, dim, labels, title):
    counts = df[dim].value_counts().reset_index()
    counts.columns = ["dim", "count"]
    counts["label"] = counts["dim"].map(labels)
    fig = px.bar(counts, x="label", y="count", color="label",
                 color_discrete_sequence=ACCENT, title=title,
                 labels={"label": "", "count": "Users"})
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    return fig


# ──────────────────────────────────────────────
# Sidebar filters
# ──────────────────────────────────────────────
def build_sidebar(df):
    with st.sidebar:
        st.markdown("## 🧠 MBTI Dashboard")
        st.markdown("---")

        types = sorted(df["type"].unique())
        selected = st.multiselect("🔎 Filter by Personality Type", types, default=types)

        st.markdown("#### MBTI Dimensions")
        ie = st.radio("Energy", ["All", "Introvert (I)", "Extrovert (E)"], horizontal=True)
        ns = st.radio("Information", ["All", "Intuitive (N)", "Sensing (S)"], horizontal=True)
        tf = st.radio("Decisions", ["All", "Thinking (T)", "Feeling (F)"], horizontal=True)
        jp = st.radio("Lifestyle", ["All", "Judging (J)", "Perceiving (P)"], horizontal=True)

        st.markdown("---")
        wc_min, wc_max = int(df["Word_count"].min()), int(df["Word_count"].max())
        wc_range = st.slider("📝 Word Count Range", wc_min, wc_max, (wc_min, wc_max))

        st.markdown("---")
        st.caption("Built with Streamlit • Data: [Kaggle MBTI](https://www.kaggle.com/datasnaek/mbti-type)")

    return selected, ie, ns, tf, jp, wc_range


def apply_filters(df, selected_types, ie, ns, tf, jp, wc_range):
    """Narrow the dataframe based on sidebar choices."""
    out = df[df["type"].isin(selected_types)]

    # extract the letter from strings like "Introvert (I)"
    dim_filters = {"IE": ie, "NS": ns, "TF": tf, "JP": jp}
    for col, val in dim_filters.items():
        if val != "All":
            letter = val[val.index("(") + 1]
            out = out[out[col] == letter]

    out = out[out["Word_count"].between(*wc_range)]
    return out


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    df = load_data()
    selected, ie, ns, tf, jp, wc_range = build_sidebar(df)
    data = apply_filters(df, selected, ie, ns, tf, jp, wc_range)

    # header
    st.markdown("# 🧠 MBTI Personality Dashboard")
    st.markdown(
        "> Explore how personality types shape writing style, sentiment, "
        "and word usage — based on the **Myers-Briggs (MBTI)** dataset."
    )

    # quick stats row
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users", f"{len(data):,}")
    c2.metric("Types", data["type"].nunique())
    c3.metric("Avg Words", f"{data['Word_count'].mean():,.0f}")
    c4.metric("Avg Sentiment", f"{data['sentiment'].mean():.3f}")
    c5.metric("Lexical Diversity", f"{data['lexical_diversity'].mean():.3f}")

    # ── tabs ──
    tab_dist, tab_posts, tab_sent, tab_wc, tab_dims, tab_ml, tab_explore = st.tabs(
        ["📊 Type Distribution", "📝 Post Analysis", "💬 Sentiment",
         "☁️ Word Cloud", "🔗 Dimensions", "🤖 MBTI Prediction", "🔍 Explorer"]
    )

    # --- distribution ---
    with tab_dist:
        st.subheader("Personality Type Distribution")
        st.caption("How are the 16 MBTI types represented in the dataset?")
        left, right = st.columns([3, 2])
        left.plotly_chart(type_distribution_chart(data), width="stretch")
        right.plotly_chart(ie_pie(data), width="stretch")
        st.info(
            "💡 The dataset skews heavily toward **Introverted** and **Intuitive** types "
            "— likely because the source is an online personality forum."
        )

    # --- post analysis ---
    with tab_posts:
        st.subheader("Post Length & Writing Style")
        left, right = st.columns(2)
        left.plotly_chart(wordcount_hist(data), width="stretch")
        right.plotly_chart(wordcount_box(data), width="stretch")
        st.plotly_chart(
            avg_feature_chart(data, "avg_word_length", "Average Word Length by Type"),
            width="stretch",
        )
        st.info(
            "💡 Introverted types tend to write longer posts. "
            "Average word length stays fairly consistent across types."
        )

    # --- sentiment ---
    with tab_sent:
        st.subheader("Sentiment Analysis")
        st.caption("TextBlob polarity: −1 (negative) → 0 (neutral) → +1 (positive)")
        left, right = st.columns(2)
        left.plotly_chart(sentiment_by_type(data), width="stretch")
        right.plotly_chart(tf_sentiment_violin(data), width="stretch")
        st.plotly_chart(correlation_heatmap(data), width="stretch")
        st.info(
            "💡 Feeling (F) types write more positively than Thinking (T) types. "
            "Most types show slightly positive sentiment overall."
        )

    # --- word cloud ---
    with tab_wc:
        st.subheader("Word Cloud")
        st.caption("Most frequent words (stop-words removed).")
        options = ["All Selected Types"] + sorted(data["type"].unique())
        pick = st.selectbox("Generate for:", options)
        subset = data if pick == "All Selected Types" else data[data["type"] == pick]
        if subset.empty:
            st.warning("No data — adjust the sidebar filters.")
        else:
            fig = make_wordcloud(subset)
            st.pyplot(fig)
            plt.close(fig)

    # --- dimensions ---
    with tab_dims:
        st.subheader("MBTI Dimension Comparisons")

        dims = [
            ("IE", {"I": "Introvert", "E": "Extrovert"}, "Introvert vs Extrovert"),
            ("NS", {"N": "Intuitive", "S": "Sensing"}, "Intuitive vs Sensing"),
            ("TF", {"T": "Thinking", "F": "Feeling"}, "Thinking vs Feeling"),
            ("JP", {"J": "Judging", "P": "Perceiving"}, "Judging vs Perceiving"),
        ]
        row = st.columns(2)
        for i, (col, labels, title) in enumerate(dims):
            row[i % 2].plotly_chart(dimension_bar(data, col, labels, title), width="stretch")

        # cross-dimension feature comparison
        st.markdown("---")
        feature = st.selectbox("Compare a feature across dimensions:",
                               ["Word_count", "avg_word_length", "lexical_diversity",
                                "self_reference_ratio", "sentiment", "punctuation_count", "link_count"])
        cols = st.columns(4)
        for i, dim in enumerate(DIMENSIONS):
            avg = data.groupby(dim)[feature].mean().reset_index()
            fig = px.bar(avg, x=dim, y=feature, color=dim,
                         color_discrete_sequence=ACCENT, title=f"{feature} by {dim}")
            fig.update_layout(**CHART_LAYOUT, showlegend=False, height=350)
            cols[i].plotly_chart(fig, width="stretch")

    # --- ML prediction demo ---
    with tab_ml:
        st.subheader("MBTI Prediction Demo")
        st.caption(
            "A Logistic Regression classifier is trained on TF-IDF features for each "
            "MBTI dimension (I/E, N/S, T/F, J/P). Click below to see it in action."
        )

        # train models (cached after first run)
        models, vectorizers, test_sets = train_models(df)

        if st.button("🎲 Generate Random Sample Prediction", type="primary"):
            # pick a random sample from the test set
            dim_key = "IE"  # use IE test set for sampling
            test_idx = test_sets[dim_key]
            random_idx = random.choice(list(test_idx))

            sample_text = df.loc[random_idx, "posts"]
            actual_type = df.loc[random_idx, "type"]
            predicted_type, confidence = predict_mbti(sample_text, models, vectorizers)

            # show text preview
            st.markdown("#### 📄 Sample Text")
            st.text_area("Post preview", sample_text[:500] + "…", height=140, disabled=True)

            # show actual vs predicted side by side
            st.markdown("#### 🎯 Result")
            col_actual, col_pred, col_match = st.columns(3)
            col_actual.metric("Actual MBTI", actual_type)
            col_pred.metric("Predicted MBTI", predicted_type)
            match = actual_type == predicted_type
            col_match.metric("Full Match", "✅ Yes" if match else "❌ No")

            # per-dimension breakdown
            st.markdown("#### 📊 Dimension Confidence")
            dim_cols = st.columns(4)
            dim_labels = {
                "IE": ("Introvert / Extrovert", "I", "E"),
                "NS": ("Intuitive / Sensing", "N", "S"),
                "TF": ("Thinking / Feeling", "T", "F"),
                "JP": ("Judging / Perceiving", "J", "P"),
            }
            for i, dim in enumerate(DIMENSIONS):
                pred_letter, prob = confidence[dim]
                actual_letter = actual_type[i]
                is_correct = pred_letter == actual_letter
                label, opt_a, opt_b = dim_labels[dim]

                with dim_cols[i]:
                    st.markdown(f"**{label}**")
                    st.metric(
                        f"Predicted: {pred_letter}",
                        f"{prob * 100:.1f}%",
                        delta="Correct" if is_correct else "Wrong",
                        delta_color="normal" if is_correct else "inverse",
                    )
                    st.progress(prob)

            # overall accuracy note
            correct_dims = sum(
                1 for i, dim in enumerate(DIMENSIONS)
                if confidence[dim][0] == actual_type[i]
            )
            st.info(
                f"💡 The model predicted **{correct_dims}/4** dimensions correctly for this sample. "
                f"Confidence values show the model's probability for its chosen label."
            )
        else:
            st.markdown(
                '<div style="text-align:center; padding:60px 0; color:#888;">'
                '👆 Click the button above to generate a random prediction'
                '</div>',
                unsafe_allow_html=True,
            )

    # --- explorer ---
    with tab_explore:
        st.subheader("Interactive Data Explorer")
        default_cols = ["type", "IE", "NS", "TF", "JP", "Word_count", "sentiment"]
        show = st.multiselect("Columns to display", data.columns.tolist(), default=default_cols)
        st.dataframe(data[show] if show else data, width="stretch", height=500)
        st.download_button("⬇️ Download filtered CSV",
                           data.to_csv(index=False).encode(),
                           file_name="mbti_filtered.csv", mime="text/csv")


if __name__ == "__main__":
    main()
