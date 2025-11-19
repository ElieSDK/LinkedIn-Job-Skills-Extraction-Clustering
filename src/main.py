import os, re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from google.colab import drive

# settings
ZIP_FILE_PATH = '/drive/linkedin.zip'
UNZIP_DIR = 'drive/linkedin_unzipped'
POSTINGS_FILE = f'{UNZIP_DIR}/postings.csv'
SAMPLE_SIZE = 500

# basic list of skills (not exhaustive)
TECH_SKILLS = [
    'python','java','sql','aws','tensorflow','machine learning','data science',
    'javascript','react','node.js','c++','c#','php','ruby','go','swift','kotlin',
    'docker','kubernetes','azure','gcp','cloud','linux','unix','git','tableau',
    'power bi','excel','spark','hadoop','scala','r','agile','scrum','jira',
    'confluence','api','frontend','backend','fullstack','devops','cybersecurity'
]

NORMALIZATION_MAP = {
    'ml':'machine learning','ds':'data science','js':'javascript','nodejs':'node.js',
    'cpp':'c++','csharp':'c#','golang':'go','powerbi':'power bi'
}


def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab/english.pickle')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def load_data():
    drive.mount('/content/drive')

    if not os.path.exists(UNZIP_DIR):
        os.makedirs(UNZIP_DIR)

    print("unzipping...")
    !unzip -qqo "{ZIP_FILE_PATH}" -d "{UNZIP_DIR}"

    df = pd.read_csv(POSTINGS_FILE)
    df = df.head(SAMPLE_SIZE).copy()  # small sample for speed

    print("sample loaded:", len(df))
    return df


def clean(text):
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)

    tokens = word_tokenize(t)
    sw = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in sw]

    return " ".join(tokens)


def find_skills(t):
    out = []
    for s in TECH_SKILLS:
        if re.search(r'\b'+re.escape(s)+r'\b', t):
            out.append(s)
    return list(set(out))


def normalize(sk_list):
    out = []
    for s in sk_list:
        s2 = NORMALIZATION_MAP.get(s.lower(), s.lower())
        out.append(s2)
    return sorted(list(set(out)))


def extract_skills(df):
    print("cleaning & extracting skills...")

    df['clean_text'] = df['description'].apply(clean)
    df['skills'] = df['clean_text'].apply(find_skills)
    df['skills'] = df['skills'].apply(normalize)

    return df


def cluster_jobs(df, k=8):
    print("clustering...")

    df['skills_str'] = df['skills'].apply(lambda x: " ".join(x))

    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(df['skills_str'])

    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['cluster'] = model.fit_predict(X)

    print(df['cluster'].value_counts())
    return df


def analyze_salary(df):
    print("\nSalary vs skills... (if salary exists)")

    if 'normalized_salary' not in df.columns:
        print("no salary field found")
        return

    s = df[df['normalized_salary'].notnull()]
    if s.empty:
        print("no salary data in sample")
        return

    avg = s.groupby('cluster')['normalized_salary'].mean().sort_values(ascending=False)
    print("\nAvg salary per cluster:")
    print(avg)

    # top skills
    for c in s['cluster'].unique():
        subset = s[s['cluster']==c]
        skills = [x for lst in subset['skills'] for x in lst]
        counts = Counter(skills).most_common(5)
        print(f"\nCluster {c} (avg {avg.get(c,'?')}):")
        for skill, cnt in counts:
            print(f"  {skill}: {cnt}")


if __name__ == "__main__":
    setup_nltk()
    df = load_data()
    df = extract_skills(df)
    df = cluster_jobs(df, 8)
    analyze_salary(df)
    print("done.")
