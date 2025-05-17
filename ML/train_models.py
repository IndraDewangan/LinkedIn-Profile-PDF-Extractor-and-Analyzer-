import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import spacy
import random
from faker import Faker

#CATEGORY PREDICTION
# Load dataset from JSON file
with open("./dataset.json", "r") as file:
    data = json.load(file)

# Extract job titles and categories
X_train = [item["title"] for item in data['jobs']]
y_train = [item["category"] for item in data['jobs']]

# Convert text into numerical representation
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
with open("./category_predict/profile_classifier_large.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("./category_predict/vectorizer_large.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ CATEGORY PREDICTIO-Model trained successfully")

# JOB RECOMMENDATION
# Load dataset from JSON file
with open("./dataset.json", "r") as json_file:
    job_data = json.load(json_file)

# Extract job titles and skills
job_titles = [job["job_title"] for job in job_data["job_database"]]
job_skills = [" ".join(job["skills"]) for job in job_data["job_database"]]

# Convert skills into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(job_skills)

# Save the trained model and vectorizer
with open("./job_recommend/job_recommendation_model.pkl", "wb") as model_file:
    pickle.dump(job_vectors, model_file)

with open("./job_recommend/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open("./job_recommend/job_titles.pkl", "wb") as titles_file:
    pickle.dump(job_titles, titles_file)

print("✅ JOB RECOMMENDATION-Model training complete!")

#SKILL RECOMMENDATION
# Step 1: Load Skill Database from JSON
with open("./dataset.json", "r") as file:
    data = json.load(file)
    skill_database = data["skills"]

# Step 2: Train TF-IDF Model
vectorizer = TfidfVectorizer()
skill_vecs = vectorizer.fit_transform(skill_database)

# Step 3: Save Model
with open("./skill_recommend/skill_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("./skill_recommend/skill_vectors.pkl", "wb") as vecs_file:
    pickle.dump(skill_vecs, vecs_file)

print("✅ SKILL RECOMMENDATION model saved successfully!")

#Headline analyser
# Initialize tools
nlp = spacy.load("en_core_web_sm")
fake = Faker()

# Sample dataset with all required scores
data = pd.DataFrame({
    'headline': [
        # Existing good headlines (10)
        "DevOps Architect | AWS | CI/CD | Ansible | Docker",
        "Senior Data Analyst | SQL | Power BI | Python",
        "AI Engineer | Computer Vision | Deep Learning | TensorFlow",
        "Python Developer | Django | FastAPI | PostgreSQL",
        "ML Engineer | Scikit-Learn | XGBoost | NLP",
        "Cloud Consultant | GCP | Kubernetes | Cloud Migrations",
        "Data Engineer | PySpark | Big Data | AWS Glue",
        "Software Engineer | C++ | Systems Programming | Linux",
        "Solutions Architect | AWS Certified | Enterprise Cloud",
        "BI Developer | SSIS | Tableau | Data Warehousing",
        
        # New good headlines (30)
        "Frontend Developer | React | TypeScript | Redux | UI/UX",
        "Backend Engineer | Node.js | Microservices | REST APIs",
        "Data Scientist | Machine Learning | Statistical Modeling",
        "Security Engineer | Ethical Hacking | Penetration Testing",
        "Product Manager | Agile | Scrum | User Stories | Roadmaps",
        "UX Designer | Figma | Prototyping | User Research",
        "Blockchain Developer | Solidity | Smart Contracts | Web3",
        "iOS Developer | Swift | SwiftUI | Mobile App Development",
        "Android Developer | Kotlin | Jetpack Compose | MVVM",
        "Cloud Architect | Azure | Terraform | Infrastructure as Code",
        "Full Stack Developer | MERN Stack | GraphQL | Next.js",
        "Data Analyst | Excel | Power BI | Data Visualization",
        "QA Engineer | Automation Testing | Selenium | TestNG",
        "DevOps Engineer | Jenkins | GitLab CI | Infrastructure",
        "Technical Writer | API Documentation | Markdown | Git",
        "Machine Learning Engineer | NLP | Transformers | PyTorch",
        "Database Administrator | SQL Server | Oracle | NoSQL",
        "Network Engineer | CCNA | Firewalls | VPN | Troubleshooting",
        "Digital Marketing Specialist | SEO | PPC | Analytics",
        "Salesforce Developer | Apex | Lightning Components | LWC",
        "Game Developer | Unity | C# | 3D Modeling | Shaders",
        "Embedded Systems Engineer | C | RTOS | Microcontrollers",
        "Technical Product Manager | APIs | SDKs | Developer Tools",
        "Site Reliability Engineer | Monitoring | Alerting | SLOs",
        "Computer Vision Engineer | OpenCV | YOLO | Image Processing",
        "Business Analyst | Requirements Gathering | Process Mapping",
        "Technical Lead | Architecture | Code Reviews | Mentoring",
        "Chief Technology Officer | Tech Strategy | Team Building",
        "Data Governance Specialist | GDPR | Compliance | Security",
        "AR/VR Developer | Unity3D | Unreal Engine | 3D Math",
        
        # Existing weak headlines (10)
        "Want a job in IT field",
        "Beginner learning programming",
        "Fresher with some knowledge in C++",
        "Need a job in data science",
        "Looking for ML internship",
        "Open to internship and job roles",
        "Passionate coder looking for job",
        "Interested in software jobs",
        "Hoping for job offers",
        "Hardworking individual looking to grow",
        
        # New weak headlines (20)
        "Seeking entry-level position in tech",
        "Recent graduate looking for opportunities",
        "Want to work in software company",
        "Need guidance to start IT career",
        "Looking for career change to tech",
        "Trying to learn coding skills",
        "Want to become data scientist",
        "Searching for web development job",
        "Need job in good company",
        "Looking for chance to prove myself",
        "Want to work with computers",
        "Seeking job after completing course",
        "Need stable job in IT sector",
        "Looking for growth opportunities",
        "Want to join MNC company",
        "Need work from home job",
        "Looking for good salary job",
        "Want to work in Google someday",
        "Need job to support family",
        "Looking for first job in IT"
    ],
    'keyword_score': [
        # Existing good headlines (10)
        9, 9, 9, 9, 9, 8, 8, 8, 8, 7,
        
        # New good headlines (30)
        9, 9, 9, 9, 8, 8, 9, 8, 8, 9,
        9, 8, 8, 9, 7, 9, 8, 8, 8, 9,
        8, 8, 8, 9, 9, 7, 8, 9, 7, 8,
        
        # Existing weak headlines (10)
        3, 2, 3, 3, 3, 2, 3, 3, 2, 2,
        
        # New weak headlines (20)
        3, 2, 3, 2, 3, 2, 3, 3, 2, 2,
        2, 2, 3, 2, 3, 2, 3, 2, 2, 3
    ],
    'clarity_score': [
        # Existing good headlines (10)
        9, 9, 9, 9, 8, 8, 8, 8, 8, 7,
        
        # New good headlines (30)
        9, 9, 9, 8, 9, 9, 8, 9, 9, 9,
        9, 8, 8, 9, 8, 9, 8, 8, 9, 9,
        8, 8, 9, 9, 9, 8, 9, 9, 8, 9,
        
        # Existing weak headlines (10)
        4, 5, 5, 4, 5, 4, 5, 4, 3, 4,
        
        # New weak headlines (20)
        5, 5, 4, 5, 5, 4, 5, 4, 4, 5,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5
    ],
    'structure_score': [
        # Existing good headlines (10)
        9, 9, 9, 9, 8, 8, 8, 8, 8, 7,
        
        # New good headlines (30)
        9, 9, 8, 8, 9, 9, 8, 9, 9, 9,
        9, 8, 8, 9, 7, 9, 8, 8, 9, 9,
        8, 8, 9, 9, 9, 8, 9, 9, 7, 9,
        
        # Existing weak headlines (10)
        2, 3, 4, 3, 3, 3, 4, 3, 2, 3,
        
        # New weak headlines (20)
        3, 4, 3, 3, 4, 3, 4, 3, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4
    ],
    'engagement_score': [
        # Existing good headlines (10)
        8, 8, 8, 8, 7, 7, 7, 7, 7, 6,
        
        # New good headlines (30)
        8, 8, 8, 9, 8, 9, 8, 9, 9, 9,
        9, 7, 7, 9, 7, 9, 7, 8, 9, 9,
        8, 7, 9, 9, 9, 7, 9, 9, 7, 9,
        
        # Existing weak headlines (10)
        4, 5, 5, 4, 5, 4, 6, 4, 3, 5,
        
        # New weak headlines (20)
        5, 5, 4, 5, 5, 4, 6, 4, 4, 5,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5
    ],
    'uniqueness_score': [
        # Existing good headlines (10)
        7, 6, 7, 6, 6, 6, 5, 5, 5, 5,
        
        # New good headlines (30)
        7, 6, 7, 8, 7, 8, 7, 7, 7, 7,
        7, 6, 6, 7, 6, 8, 6, 6, 7, 8,
        7, 6, 8, 8, 8, 6, 8, 8, 6, 8,
        
        # Existing weak headlines (10)
        3, 4, 4, 3, 4, 3, 5, 3, 2, 4,
        
        # New weak headlines (20)
        4, 4, 3, 4, 4, 3, 5, 3, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4
    ]
})

# Calculate overall weighted score
data['overall_score'] = (
    0.3 * data['keyword_score'] + 
    0.25 * data['clarity_score'] + 
    0.2 * data['structure_score'] + 
    0.15 * data['engagement_score'] + 
    0.1 * data['uniqueness_score']
)

def extract_features(text):
    """Extract features from headline text"""
    doc = nlp(text)
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'has_pipes': int('|' in text),
        'has_emojis': int(any(ord(c) > 127 for c in text)),
        'sentiment': TextBlob(text).sentiment.polarity,
        'noun_count': sum(1 for token in doc if token.pos_ == 'NOUN'),
        'verb_count': sum(1 for token in doc if token.pos_ == 'VERB'),
        'adj_count': sum(1 for token in doc if token.pos_ == 'ADJ'),
        'is_job_seeking': int(any(word in text.lower() for word in 
                                 ['looking', 'want', 'need', 'seeking', 'hoping']))
    }
    return features

# Extract features for all headlines
features_list = []
for headline in data['headline']:
    features_list.append(extract_features(headline))
    
features_df = pd.DataFrame(features_list)

# Combine features with original data
full_data = pd.concat([data, features_df], axis=1)

# Prepare training data
X = full_data.drop(columns=['headline', 'keyword_score', 'clarity_score', 
                          'structure_score', 'engagement_score', 
                          'uniqueness_score', 'overall_score'])
y = full_data['overall_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

# Save model and feature columns
with open("./headline_analyser/headline_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("./headline_analyser/feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model and feature columns saved successfully.")

# Additional: Save TF-IDF vectorizer for text processing
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(data['headline'])
with open("./headline_analyser/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("TF-IDF vectorizer saved.")