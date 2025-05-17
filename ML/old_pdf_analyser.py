import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
import pandas as pd
import re
from textstat import flesch_reading_ease
from collections import Counter

sumScores=0
overAllScore=0

def pdfAnalyser():
    global sumScores

    with open("./extracted_pdf_data.json", "r") as json_file:
        pdfData = json.load(json_file)
        pdfData=pdfData[0]

    #CATEGORY PREDICTION
    # Load the trained model and vectorizer
    with open("./category_predict/profile_classifier_large.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("./category_predict/vectorizer_large.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Load the dataset
    with open("./dataset.json", "r") as json_file:
        data = json.load(json_file)

    # Check if job title exists in the dataset
    def get_job_category(job_title):
        # Check if the job title is already in the dataset
        for job in data.get("jobs", []):
            if job["title"] == job_title:
                return job["category"]
        
        # If not found, predict using the model
        job_title_vec = vectorizer.transform([job_title])  # Convert text to numerical form
        predicted_category = model.predict(job_title_vec)  # Predict the category
        
        # Add the new job title and category to the dataset
        new_job = {"title": job_title, "category": predicted_category[0]+'(Self_Learned)'}
        data["jobs"].append(new_job)
        
        # Save the updated dataset back to the JSON file
        with open("./dataset.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
            print("Self Learning Activated")
        
        return predicted_category[0]

    # Example usage
    job_title = pdfData["heading"]
    predicted_category = get_job_category(job_title)

    print(f"Predicted category: {predicted_category}")

    #JOB RECOMMENDATION
    # Load the trained model, vectorizer, and job titles
    with open("./job_recommend/job_recommendation_model.pkl", "rb") as model_file:
        job_vectors = pickle.load(model_file)

    with open("./job_recommend/vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open("./job_recommend/job_titles.pkl", "rb") as titles_file:
        job_titles = pickle.load(titles_file)

    # # Load live jobs from the JSON file
    # with open("./live_jobs_WWR.json", "r") as live_jobs_file:
    #     live_jobs = json.load(live_jobs_file)

    # Get user input skills
    user_skills = pdfData["top_skills"]

    # Convert user input into TF-IDF vector
    user_input_vector = vectorizer.transform([" ".join(user_skills)])

    # Compute cosine similarity
    similarities = cosine_similarity(user_input_vector, job_vectors).flatten()

    # Get top 10 matching jobs
    top_indices = similarities.argsort()[::-1][:10]  # Get indices of top 10 matches

    # Filter out low-matching results (threshold can be adjusted)
    recommended_jobs = [(job_titles[i], similarities[i]) for i in top_indices if similarities[i] > 0]

    # Display recommended jobs based on skills
    if recommended_jobs:
        print("\nRecommended Jobs based on Skills:")
        for idx, (job, score) in enumerate(recommended_jobs, start=1):
            print(f"{idx}. {job} (Similarity Score: {score:.2f})")

        # Track recommended live jobs to avoid duplicates
        recommended_live_jobs = set()  # Use a set to store unique live job titles

        # print("\nLive Job Recommendations from the Available Live Jobs:")
        # for live_job in live_jobs:
        #     # Check if the live job title matches any of the recommended jobs
        #     for recommended_job, _ in recommended_jobs:
        #         if recommended_job.lower() in live_job["title"].lower():
        #             # Add to set to avoid recommending the same live job multiple times
        #             if live_job["title"] not in recommended_live_jobs:
        #                 print(f"Title: {live_job['title']}, Company: {live_job['company']}")
        #                 recommended_live_jobs.add(live_job["title"])

    else:
        print("No matching jobs found. Try different skills.")

    #SKILL RECOMMENDATION
    # Step 1: Load Model
    with open("./skill_recommend/skill_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

    with open("./skill_recommend/skill_vectors.pkl", "rb") as vecs_file:
        skill_vecs = pickle.load(vecs_file)

    with open("./dataset.json", "r") as file:
        data = json.load(file)
        skill_database = data["skills"]
    # Step 2: Recommend Skills Function
    def recommend_skills(profile_text, existing_skills, skill_database):
        profile_vec = vectorizer.transform([profile_text])
        similarities = cosine_similarity(profile_vec, skill_vecs)
        top_skills = [skill_database[i] for i in similarities.argsort()[0][-5:]]
        return [skill for skill in top_skills if skill not in existing_skills]

    # Step 3: Test
    profile_text = pdfData["heading"]
    existing_skills = pdfData["top_skills"]
    recommended = recommend_skills(profile_text, existing_skills, skill_database)
    print("Recommended Skills based on top skills:", recommended)


    #adding new skill
    for skill in existing_skills:
        if skill not in skill_database:
            print(f"Adding new skill: {skill}")
            skill_database.append(skill)

    # Save the updated dataset back to the job_data.json file
    with open('./dataset.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    #HEADLINE ANALYSER
    #HEADLINE
    # Load the necessary resources
    try:
        # Load the trained model
        with open("./headline_analyser/headline_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load the feature columns
        with open("./headline_analyser/feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        
        # Load the TF-IDF vectorizer
        with open("./headline_analyser/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        # Load spaCy language model
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading resources: {e}")
        exit()

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

    def calculate_component_scores(features):
        """Calculate individual component scores based on features"""
        # Keyword Score (30% weight)
        keyword_score = min(10, max(1, 
            (features['noun_count'] * 0.8) + 
            (features['verb_count'] * 0.4) +
            (5 if features['has_pipes'] else 0)
        ))
        
        # Clarity Score (25% weight)
        clarity_score = min(10, max(1,
            3 + (features['sentiment'] * 5) +
            (features['adj_count'] * 0.3) +
            (3 if features['length'] > 50 else 0)
        ))
        
        # Structure Score (20% weight)
        structure_score = min(10, max(1,
            3 + (features['has_pipes'] * 3) +
            (features['has_emojis'] * 1.5) +
            (0.1 * features['length'])
        ))
        
        # Engagement Score (15% weight)
        engagement_score = min(10, max(1,
            3 + (features['verb_count'] * 0.8) +
            (features['sentiment'] * 3) -
            (features['is_job_seeking'] * 2)
        ))
        
        # Uniqueness Score (10% weight)
        uniqueness_score = min(10, max(1,
            5 + (features['length'] / 30) - 
            (features['word_count'] / 10) +
            (features['has_emojis'] * 1)
        ))
        
        return {
            'keyword_score': round(keyword_score, 1),
            'clarity_score': round(clarity_score, 1),
            'structure_score': round(structure_score, 1),
            'engagement_score': round(engagement_score, 1),
            'uniqueness_score': round(uniqueness_score, 1)
        }

    def analyze_headline(headline):
        """Analyze a LinkedIn headline and return scores"""
        # Extract features
        features = extract_features(headline)
        
        # Create DataFrame with same structure as training data
        df = pd.DataFrame([features])
        df = df[feature_columns]  # Ensure same column order as training
        
        # Predict overall score
        overall_score = model.predict(df)[0]
        overall_score = max(1, min(10, round(overall_score, 1)))
        
        # Calculate component scores
        component_scores = calculate_component_scores(features)
        
        return {
            'headline': headline,
            'overall_score': overall_score,
            'component_scores': component_scores,
            'features': features
        }

    def print_analysis(analysis):
        global sumScores
        """Print the analysis results in a readable format"""
        print(f"\nOverall Score: {analysis['overall_score']}/10")
        sumScores += analysis['overall_score']
        print("\nComponent Scores:")
        print(f"- Keyword Optimization (30%): {analysis['component_scores']['keyword_score']}/10")
        print(f"- Clarity & Value Proposition (25%): {analysis['component_scores']['clarity_score']}/10")
        print(f"- Structure & Readability (20%): {analysis['component_scores']['structure_score']}/10")
        print(f"- Engagement Potential (15%): {analysis['component_scores']['engagement_score']}/10")
        print(f"- Uniqueness (10%): {analysis['component_scores']['uniqueness_score']}/10")
        
        print("\nRecommendations:")
        if analysis['overall_score'] < 5:
            print("⚠️ Your headline needs significant improvement")
        elif analysis['overall_score'] < 7:
            print("🔄 Your headline is decent but could be improved")
        else:
            print("✅ Your headline is strong!")
        
        if analysis['component_scores']['keyword_score'] < 6:
            print("- Add more relevant keywords for your profession")
        if analysis['component_scores']['clarity_score'] < 6:
            print("- Make your value proposition clearer")
        if analysis['component_scores']['structure_score'] < 6:
            print("- Improve structure with separators (|) or bullet points")
        if analysis['component_scores']['engagement_score'] < 6:
            print("- Add more action-oriented language")
        if analysis['component_scores']['uniqueness_score'] < 6:
            print("- Make your headline more distinctive")

    headline = pdfData["heading"]
    analysis = analyze_headline(headline)
    print_analysis(analysis)


    #SUMMARY ANALYSER
    nlp = spacy.load("en_core_web_sm")

    class SummaryAnalyzer:
        def __init__(self):
            self.cliches = [
        "team player", "hard worker", "self-starter", "go-getter", "passionate",
        "motivated individual", "strong work ethic", "detail-oriented", "fast learner",
        "excellent communication skills", "works well under pressure", "dynamic",
        "results-driven", "proven track record", "outside the box thinker",
        "natural leader", "dedicated professional", "quick learner", "multi-tasker",
        "problem solver", "works independently", "adaptable", "goal-oriented",
        "takes initiative", "strategic thinker"
    ]
            self.buzzwords = [
        "synergy", "disrupt", "pivot", "game changer", "innovative", "value add",
        "scalable", "leverage", "streamline", "ecosystem", "deep dive", "touch base",
        "bandwidth", "low-hanging fruit", "circle back", "thought leader", "bleeding edge",
        "granular", "paradigm shift", "move the needle", "digital transformation",
        "cloud-first", "AI-powered", "hyperautomation", "growth hacking", "data-driven",
        "best-in-class", "next-gen", "future-proof", "mission-critical", "customer-centric"
    ]
            self.skill_keywords = [
        # Programming & Data
        "python", "java", "c++", "javascript", "typescript", "html", "css",
        "sql", "nosql", "mongodb", "postgresql", "mysql",
        "data analysis", "data visualization", "data science",
        "machine learning", "deep learning", "tensorflow", "keras", "scikit-learn",
        "pandas", "numpy", "matplotlib", "power bi", "excel", "tableau",

        # Cloud & DevOps
        "aws", "azure", "gcp", "cloud computing", "docker", "kubernetes",
        "jenkins", "terraform", "ci/cd", "linux", "bash scripting",

        # Web & App Dev
        "react", "vue", "node.js", "express", "django", "flask", "firebase",
        "rest api", "graphql", "wordpress", "shopify", "web design", "ui/ux",

        # Business & Marketing
        "marketing strategy", "seo", "sem", "ppc", "google ads", "email marketing",
        "social media", "facebook ads", "content marketing", "crm", "hubspot",
        "brand strategy", "growth hacking", "product management",

        # Project & Team
        "agile", "scrum", "kanban", "jira", "asana", "project management",
        "stakeholder management", "team leadership", "budget management",

        # Design & Creativity
        "graphic design", "adobe photoshop", "illustrator", "figma", "canva",
        "video editing", "motion graphics", "user experience", "wireframing",

        # Communication & Analysis
        "public speaking", "technical writing", "data storytelling",
        "business analysis", "competitive analysis"
    ]

        def analyze(self, summary):
            doc = nlp(summary)
            
            scores = {
                'length_score': self._check_length(summary),
                'special_chars_score': self._check_special_chars(summary),
                'hard_skills_score': self._check_hard_skills(summary),
                'cta_score': self._check_cta(summary),
                'readability_score': self._check_readability(summary),
                'cliches_score': self._check_cliches(summary),
                'metrics_score': self._check_metrics(summary),
                'active_voice_score': self._check_active_voice(doc),
                'sentiment_score': self._check_sentiment(summary),
                'tense_score': self._check_tense(doc),
                'spelling_score': self._check_spelling(summary)
            }
            
            weights = {
                'length_score': 0.15,
                'special_chars_score': 0.05,
                'hard_skills_score': 0.20,
                'cta_score': 0.10,
                'readability_score': 0.10,
                'cliches_score': 0.10,
                'metrics_score': 0.15,
                'active_voice_score': 0.05,
                'sentiment_score': 0.05,
                'tense_score': 0.03,
                'spelling_score': 0.02
            }
            
            overall_score = sum(scores[k] * weights[k] for k in scores)
            
            return {
                'overall_score': round(overall_score * 10, 1),  # Convert to 0-10 scale
                'component_scores': {k: round(v * 10, 1) for k,v in scores.items()},
                'feedback': self._generate_feedback(scores)
            }

        # --- Evaluation Methods ---
        def _check_length(self, text):
            """Ideal: 300-2000 characters"""
            length = len(text)
            if 500 <= length <= 1500: return 1.0
            if 300 <= length < 500 or 1500 < length <= 2000: return 0.7
            return 0.3

        def _check_special_chars(self, text):
            """Penalize excessive symbols except bullets"""
            symbol_count = len(re.findall(r'[^\w\s-]', text))
            return max(0, 1 - (symbol_count / 100))  # Allow 1 symbol per 100 chars

        def _check_hard_skills(self, text):
            """Count industry-specific skills"""
            found = sum(1 for skill in self.skill_keywords if skill in text.lower())
            return min(1.0, found / 4)  # Max score for 4+ skills

        def _check_cta(self, text):
            """Check for call-to-action phrases"""
            ctas = ["contact me", "reach out", "let's connect", "get in touch"]
            return 1.0 if any(cta in text.lower() for cta in ctas) else 0.0

        def _check_readability(self, text):
            """Flesch Reading Ease (60-80 ideal)"""
            score = flesch_reading_ease(text)
            if 60 <= score <= 80: return 1.0
            if 50 <= score < 60 or 80 < score <= 90: return 0.7
            return 0.3

        def _check_cliches(self, text):
            """Detect overused phrases"""
            text_lower = text.lower()
            cliche_count = sum(1 for phrase in self.cliches + self.buzzwords if phrase in text_lower)
            return max(0.0, 1.0 - (cliche_count * 0.2))

        def _check_metrics(self, text):
            """Count quantifiable achievements"""
            metrics = re.findall(r'\d+%|\d+\+|\$\d+', text)
            return min(1.0, len(metrics) / 2)  # Max score for 2+ metrics

        def _check_active_voice(self, doc):
            """Check passive vs. active voice ratio"""
            passive_count = sum(1 for token in doc if token.dep_ == "nsubjpass")
            total_sentences = len(list(doc.sents))
            return 1.0 if (passive_count / total_sentences) < 0.2 else 0.3

        def _check_sentiment(self, text):
            """Positive sentiment preferred"""
            return (TextBlob(text).sentiment.polarity + 1) / 2  # Convert -1..1 to 0..1

        def _check_tense(self, doc):
            """Check consistent tense (prefer present)"""
            present_verbs = sum(1 for token in doc if token.tag_ in ["VBZ", "VBP"])
            past_verbs = sum(1 for token in doc if token.tag_ in ["VBD", "VBN"])
            return 1.0 if present_verbs > past_verbs else 0.5

        def _check_spelling(self, text):
            """Basic spelling check"""
            return 1.0 if TextBlob(text).correct().lower() == text.lower() else 0.0

        def _generate_feedback(self, scores):
            feedback = []
            if scores['length_score'] < 0.7:
                feedback.append("⚠️ Adjust length (500-1500 chars ideal)")
            if scores['hard_skills_score'] < 0.7:
                feedback.append(f"⚠️ Add more hard skills (found {int(scores['hard_skills_score']*4)}/4+)")
            if scores['metrics_score'] < 0.5:
                feedback.append("⚠️ Include quantifiable achievements (e.g., 'Increased X by 30%')")
            if scores['active_voice_score'] < 1.0:
                feedback.append("⚠️ Use more active voice ('Led projects' vs 'Projects were led')")
            return feedback

    # Usage Example
    analyzer = SummaryAnalyzer()
    sample_summary = pdfData["summary"]
    result = analyzer.analyze(sample_summary)

    print(f"Overall Score: {result['overall_score']}/10")
    sumScores +=result['overall_score']
    print("Component Scores:")
    for k, v in result['component_scores'].items():
        print(f"- {k.replace('_', ' ').title()}: {v}/10")
    print("Feedback:", result['feedback'])

    # EXPERIENCE ANALYSER
    nlp = spacy.load("en_core_web_sm")

    class ExperienceAnalyzer:
        def __init__(self):
            self.red_flags = ["unemployed", "gap", "fired", "laid off"]
            self.cliches = ["team player", "passionate", "hard worker", "think outside the box"]
            self.skill_keywords = ["python", "sql", "marketing", "sales", "engineering"]

        def analyze_experience(self, title, description):
            title_doc = nlp(title)
            desc_doc = nlp(description)

            scores = {
                'title_hard_skills': self._check_title_skills(title),
                'title_specificity': self._check_title_specificity(title),
                'description_detail': self._check_description_detail(description),
                'red_flags': self._check_red_flags(description),
                'quantified_impact': self._check_quantified_impact(description),
                'weak_language': self._check_weak_language(description),
                'spelling': self._check_spelling(description),
                'active_voice': self._check_active_voice(desc_doc),
                'special_chars': self._check_special_chars(description),
                'cliches': self._check_cliches(description)
            }

            # Updated weights for short experiences
            weights = {
                'title_hard_skills': 0.25,
                'title_specificity': 0.20,
                'description_detail': 0.10,
                'red_flags': 0.10,
                'quantified_impact': 0.10,
                'weak_language': 0.05,
                'spelling': 0.05,
                'active_voice': 0.05,
                'special_chars': 0.05,
                'cliches': 0.05
            }

            raw_score = sum(scores[k] * weights[k] for k in scores)
            final_score = max(0, min(10, round(raw_score * 10)))

            return {
                'score': final_score,
                'metrics': {k: round(v * 10, 1) for k, v in scores.items()},
                'feedback': self._generate_feedback(scores)
            }

        # --- Evaluation Methods ---

        def _check_title_skills(self, title):
            found = sum(1 for skill in self.skill_keywords if skill in title.lower())
            return min(1.0, found / 2)

        def _check_title_specificity(self, title):
            generic_titles = ["associate", "staff", "member", "representative"]
            return 0.0 if any(word in title.lower() for word in generic_titles) else 1.0

        def _check_description_detail(self, text):
            """For short descriptions, be more forgiving."""
            word_count = len(text.split())
            if word_count >= 5:
                return 1.0
            elif word_count >= 3:
                return 0.7
            else:
                return 0.5

        def _check_red_flags(self, text):
            return 0.0 if any(flag in text.lower() for flag in self.red_flags) else 1.0

        def _check_quantified_impact(self, text):
            metrics = re.findall(r'\d+%|\d+\+|\$\d+', text)
            if len(metrics) > 0:
                return 1.0
            else:
                return 0.7  # Instead of 0.3, be more lenient for simple titles

        def _check_weak_language(self, text):
            weak_verbs = ["helped", "assisted", "tried", "attempted"]
            return 0.0 if any(verb in text.lower() for verb in weak_verbs) else 1.0

        def _check_spelling(self, text):
            return 1.0 if TextBlob(text).correct().lower() == text.lower() else 0.0

        def _check_active_voice(self, doc):
            passive = sum(1 for token in doc if token.dep_ == "nsubjpass")
            return 1.0 if passive < 2 else 0.7  # More relaxed

        def _check_special_chars(self, text):
            bad_chars = ["!!", "??", "::"]
            return 0.0 if any(char in text for char in bad_chars) else 1.0

        def _check_cliches(self, text):
            return 0.0 if any(cliche in text.lower() for cliche in self.cliches) else 1.0

        def _generate_feedback(self, scores):
            feedback = []
            if scores['title_hard_skills'] < 0.7:
                feedback.append("⚠️ Add relevant technical/industry skills to job title if possible.")
            if scores['quantified_impact'] < 0.7:
                feedback.append("⚠️ Try mentioning any achievements or measurable impact.")
            if scores['red_flags'] < 1.0:
                feedback.append("⚠️ Remove negative language about employment gaps.")
            return feedback

    # Usage Example
    analyzer = ExperienceAnalyzer()
    expScore=0
    for i in range(len(pdfData["experience"])):
        experience = {
            "title": pdfData["experience"][i]["title"],
            "description": pdfData["experience"][i]["description"],
        }

        print(f"Experience {i+1}")
        result= analyzer.analyze_experience(**experience)
        print(f"Total Score: {result['score']}/10")
        expScore += result['score']
        print("Component Scores:")
        for k, v in result['metrics'].items():
            print(f"- {k.replace('_', ' ').title()}: {v}/10")
        print("Feedback:")
        for f in result['feedback']:
            print(f)
    expScore = expScore/len(pdfData["experience"])

    sumScores += expScore

    # EDUCATION ANALYSER
    class EducationAnalyzer:
        def __init__(self):
            self.high_degrees = ['phd', 'doctorate', 'msc', 'ms', 'mtech', 'mba', 'md', 'jd', 'mca','masters', 'postgraduate', 'post doc', 'llm', 'med']
            self.mid_degrees = ['bsc', 'bs', 'btech', 'ba', 'beng', 'bba', 'bcom', 'bca', 'undergraduate','bachelors', 'b.e', 'b.ed']
            self.low_degrees = ['diploma', 'associate', 'high school', 'secondary school', 'hsc','intermediate', 'ged', '10th', '12th', 'vocational training']
            self.top_institutes = [
        # Ivy League & Top US
        'harvard', 'stanford', 'mit', 'princeton', 'yale', 'columbia', 'upenn', 'uchicago', 'caltech', 'berkeley', 'nyu', 'cornell',

        # Top UK
        'oxford', 'cambridge', 'imperial college', 'lse', 'ucl', 'king\'s college london',

        # Top Europe
        'eth zurich', 'epfl', 'hec paris', 'insead', 'university of amsterdam', 'tu munich',

        # India
        'iit', 'iit delhi', 'iit bombay', 'iit kanpur', 'iit madras', 'iim', 'iim ahmedabad', 'bits pilani', 'iiit hyderabad', 'nits', 'du', 'jnu',

        # Asia
        'nus', 'ntu', 'university of tokyo', 'tsinghua university', 'pku', 'kaist', 'hkust',

        # Australia
        'university of melbourne', 'university of sydney', 'unsw', 'australian national university',

        # Canada
        'university of toronto', 'mcgill', 'ubc', 'waterloo',

        # Other well known
        'carnegie mellon', 'georgia tech', 'ucla', 'university of michigan', 'purdue university']

        def analyze(self, degree, institute):
            scores = {
                'degree_level': self._score_degree(degree),
                'institute_rank': self._score_institute(institute)
            }

            weights = {
                'degree_level': 0.6,
                'institute_rank': 0.4
            }

            raw_score = sum(scores[k] * weights[k] for k in scores)
            final_score = max(0, min(10, round(raw_score * 10)))

            return {
                'score': final_score,
                'component_scores': {k: round(v * 10, 1) for k, v in scores.items()},
                'feedback': self._generate_feedback(scores)
            }

        def _score_degree(self, degree):
            degree = degree.lower()
            if any(d in degree for d in self.high_degrees): return 1.0
            if any(d in degree for d in self.mid_degrees): return 0.7
            if any(d in degree for d in self.low_degrees): return 0.4
            return 0.2

        def _score_institute(self, institute):
            return 1.0 if any(name in institute.lower() for name in self.top_institutes) else 0.5

        def _generate_feedback(self, scores):
            feedback = []
            if scores['degree_level'] < 0.7:
                feedback.append("⚠️ Consider pursuing a higher-level degree to boost credibility.")
            else:
                feedback.append("✅ Great academic qualification.")

            if scores['institute_rank'] < 1.0:
                feedback.append("📌 Your institute is not recognized among top-tier schools. Highlight certifications or projects.")
            else:
                feedback.append("✅ Recognized institute strengthens your profile.")
            return feedback


    analyzer = EducationAnalyzer()
    eduScore=0
    for i in range(len(pdfData["education"])):
        result = analyzer.analyze(
            degree=pdfData["education"][i]["degree"],
            institute=pdfData["education"][i]["university"]
        )

        print("Total Score:", result['score'])
        eduScore += result['score']
        # print("Component Scores:")
        # for k, v in result['component_scores'].items():
        #     print(f"- {k.replace('_', ' ').title()}: {v}/10")
        # print("Feedback:")
        # for f in result['feedback']:
        #     print(f)
    eduScore = eduScore/len(pdfData["education"])

    sumScores += eduScore

    # OTHERS ANALYSIS
    otherScore=0
    if pdfData["linkedin_url"] !="":
        print("found Honors-Awards section")
        otherScore +=1
    else:
        print("No Honors-Awards section")
    if pdfData["Certification"] !="":
        print("found Certifications section")
        otherScore +=1
    else:
        print("No Certifications section")
    if pdfData["linkedin_url"] !="":
        print("found public URL")
        otherScore +=1
    else:
        print("No public URL")
    if pdfData["location"] !="":
        print("found location")
        otherScore +=1
    else:
        print("No public URL")
    if pdfData["top_skills"] !="":
        print("found skill section")
        otherScore +=1
    else:
        print("No skills URL")
    
    otherScore = 2*otherScore
    sumScores += otherScore
    overAllScore = int(10*(sumScores/5))
    print(f"overAll score is : {overAllScore}/100")


pdfAnalyser()
