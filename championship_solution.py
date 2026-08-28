#!/usr/bin/env python3
"""
Championship Solution for NLP with Disaster Tweets
State-of-the-art techniques aimed at top leaderboard performance
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries (handle gracefully if not available)
try:
    from sklearn.svm import LinearSVC
    HAS_SVC = True
except ImportError:
    HAS_SVC = False


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def championship_text_cleaning(text):
    """
    Championship-level text cleaning that preserves disaster signals
    """
    if pd.isna(text):
        return "", ""
    
    original = str(text)
    
    # Preserve case information for feature extraction
    has_uppercase = any(c.isupper() for c in original)
    all_uppercase = original.isupper() and len(original) > 1
    
    # Convert to lowercase for processing
    text = original.lower()
    
    # Remove URLs but track them
    url_count = len(re.findall(r'http\S+|www\S+|https\S+', text))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Track mentions but remove them
    mention_count = len(re.findall(r'@\w+', text))
    text = re.sub(r'@\w+', '', text)
    
    # Extract and preserve hashtags (important for disasters)
    hashtags = re.findall(r'#(\w+)', text)
    hashtag_count = len(hashtags)
    # Keep hashtag text without # for better word processing
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation but keep track of specific types
    exclamation_count = original.count('!')
    question_count = original.count('?')
    
    # Remove punctuation for cleaner text processing
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return cleaned text and features dictionary
    features = {
        'url_count': url_count,
        'mention_count': mention_count,
        'hashtag_count': hashtag_count,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'has_uppercase': int(has_uppercase),
        'all_uppercase': int(all_uppercase),
        'original_length': len(original),
        'has_numbers': int(bool(re.search(r'\d', original))),
        'digit_count': len(re.findall(r'\d', original))
    }
    
    return text, features

def extract_championship_features(text_series):
    """
    Extract championship-level linguistic and disaster-specific features
    """
    features_list = []
    
    for text in text_series:
        cleaned_text, feat_dict = championship_text_cleaning(text)
        
        # Basic text statistics
        feat_dict['word_count'] = len(cleaned_text.split()) if cleaned_text else 0
        feat_dict['avg_word_length'] = (
            sum(len(word) for word in cleaned_text.split()) / 
            max(len(cleaned_text.split()), 1)
        ) if cleaned_text else 0
        feat_dict['char_count'] = len(cleaned_text)
        
        # Linguistic features from original text
        original_uppercase_words = len(re.findall(r'\b[A-Z]{2,}\b', str(text)))
        feat_dict['uppercase_word_count'] = original_uppercase_words
        feat_dict['title_case_words'] = len(re.findall(r'\b[A-Z][a-z]+\b', str(text)))
        feat_dict['capitalization_ratio'] = (
            original_uppercase_words / max(len(str(text).split()), 1)
        )
        
        # Disaster-specific keyword features (expanded list)
        disaster_terms = {
            # Natural disasters
            'earthquake': 2, 'flood': 2, 'hurricane': 2, 'tsunami': 2, 
            'wildfire': 2, 'fire': 1, 'tornado': 2, 'storm': 1,
            'blizzard': 2, 'drought': 2, 'volcano': 2, 'eruption': 2,
            'landslide': 2, 'avalanche': 2, 'typhoon': 2, 'cyclone': 2,
            
            # Human-caused disasters
            'explosion': 2, 'blast': 2, 'crash': 2, 'accident': 1,
            'shooting': 2, 'attack': 2, 'bomb': 2, 'fire': 1,
            
            # Impact and response
            'evacuate': 2, 'rescue': 2, 'casualty': 2, 'injury': 1,
            'death': 2, 'dead': 2, 'die': 1, 'killed': 2,
            'missing': 2, 'trapped': 2, 'emergency': 2, 'disaster': 2,
            'crisis': 2, 'help': 1, 'please': 1, 'urgent': 2,
            'warning': 2, 'alert': 2, 'danger': 2, 'safe': 1,
            'shelter': 2, 'aid': 2, 'relief': 2, 'medical': 1,
            'hospital': 1, 'ambulance': 2, 'police': 1, 'firefighter': 2
        }
        
        disaster_score = 0
        text_lower = str(text).lower()
        for term, weight in disaster_terms.items():
            disaster_score += text_lower.count(term) * weight
        
        feat_dict['disaster_keyword_score'] = disaster_score
        feat_dict['has_disaster_keywords'] = int(disaster_score > 0)
        
        # Urgency and emotion features
        urgency_terms = ['help', 'please', 'urgent', 'emergency', 'rescue', 
                        'evacuate', 'asap', 'now', 'immediately', 'fast']
        urgency_score = sum(text_lower.count(term) for term in urgency_terms)
        feat_dict['urgency_score'] = urgency_score
        
        # Question and excitement combinations
        feat_dict['exclamation_to_question'] = (
            feat_dict['exclamation_count'] / max(feat_dict['question_count'], 1)
        )
        
        # Social media engagement indicators
        feat_dict['engagement_potential'] = (
            feat_dict['hashtag_count'] * 2 + 
            feat_dict['mention_count'] + 
            feat_dict['url_count']
        )
        
        # Readability and complexity (simplified)
        if feat_dict['word_count'] > 0:
            feat_dict['lexical_diversity'] = len(set(cleaned_text.split())) / feat_dict['word_count']
            feat_dict['avg_sentence_length'] = feat_dict['word_count'] / max(feat_dict['question_count'] + feat_dict['exclamation_count'] + 1, 1)
        else:
            feat_dict['lexical_diversity'] = 0
            feat_dict['avg_sentence_length'] = 0
            
        features_list.append(feat_dict)
    
    return pd.DataFrame(features_list), [feat_dict['cleaned_text'] for feat_dict in [{'cleaned_text': championship_text_cleaning(t)[0]} for t in text_series]]

def create_championship_models():
    """
    Create a diverse set of championship-level models
    """
    models = []
    
    # Different Logistic Regression variants
    models.append(('lr_l1_lbfgs', LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=0.1,
        max_iter=1000,
        random_state=42
    )))
    
    models.append(('lr_l2_lbfgs', LogisticRegression(
        penalty='l2', 
        solver='lbfgs', 
        C=1.0,
        max_iter=1000,
        random_state=42
    )))
    
    models.append(('lr_l2_saga', LogisticRegression(
        penalty='l2', 
        solver='saga', 
        C=10.0,
        max_iter=1000,
        random_state=42
    )))
    
    # Naive Bayes with different alpha values
    models.append(('nb_low_alpha', MultinomialNB(alpha=0.01)))
    models.append(('nb_medium_alpha', MultinomialNB(alpha=0.1)))
    models.append(('nb_high_alpha', MultinomialNB(alpha=1.0)))
    
    # Tree-based models (if available)
    if HAS_XGB:
        models.append(('xgboost', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )))
    
    if HAS_LGBM:
        models.append(('lightgbm', LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )))
    
    # SVM if available
    if HAS_SVC:
        models.append(('linear_svc', LinearSVC(
            C=1.0,
            random_state=42,
            max_iter=2000
        )))
    
    return models

def create_championship_ensemble(X_train, y_train):
    """
    Create championship-level ensemble using stacking
    """
    from scipy.sparse import hstack
    
    print("🔧 Creating championship-level features...")
    
    # Create enhanced TF-IDF vectorizer
    text_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 4),  # Up to 4-grams for better context
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2',
        use_idf=True
    )
    
    # Transform text data
    print("   Vectorizing text...")
    X_train_text = text_vectorizer.fit_transform(X_train)
    
    # Extract championship features
    print("   Extracting championship features...")
    X_train_features, cleaned_texts = extract_championship_features(X_train)
    
    # Combine features
    print("   Combining features...")
    X_train_combined = hstack([X_train_text, X_train_features.values])
    
    # Create base models
    print("   Creating base models...")
    base_models = create_championship_models()
    
    # Create meta-learner
    meta_learner = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking classifier
    print("   Building stacking ensemble...")
    # For models that don't support predict_proba, we'll use those that do
    estimators_for_stacking = []
    for name, model in base_models:
        # Check if model supports predict_proba (needed for stacking)
        if hasattr(model, 'predict_proba') or 'LinearSVC' not in str(type(model)):
            estimators_for_stacking.append((name, model))
    
    # If we don't have enough models that support predict_proba, fall back to voting
    if len(estimators_for_stacking) >= 3:
        print(f"   Using stacking with {len(estimators_for_stacking)} models...")
        ensemble = StackingClassifier(
            estimators=estimators_for_stacking,
            final_estimator=meta_learner,
            cv=5,
            stack_method='auto',
            n_jobs=-1
        )
    else:
        print("   Falling back to voting ensemble...")
        # Use voting classifier with models that work well together
        voting_models = [
            ('lr_l1', LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)),
            ('lr_l2', LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, random_state=42)),
            ('nb', MultinomialNB(alpha=0.1))
        ]
        
        if HAS_XGB:
            voting_models.append(('xgb', XGBClassifier(n_estimators=100, max_depth=4, random_state=42)))
        if HAS_LGBM:
            voting_models.append(('lgbm', LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)))
        
        # Try soft voting first
        try:
            ensemble = VotingClassifier(
                estimators=voting_models,
                voting='soft',
                n_jobs=-1
            )
        except:
            # Fall back to hard voting
            ensemble = VotingClassifier(
                estimators=voting_models,
                voting='hard',
                n_jobs=-1
            )
    
    # Train the ensemble
    print("   Training ensemble...")
    ensemble.fit(X_train_combined, y_train)
    
    return ensemble, text_vectorizer, X_train_features.columns.tolist()

def predict_championship(model, vectorizer, feature_columns, X_test):
    """
    Generate predictions using the championship model
    """
    from scipy.sparse import hstack
    
    print("🔮 Generating championship predictions...")
    
    # Transform test text
    X_test_text = vectorizer.transform(X_test)
    
    # Extract championship features for test data
    X_test_features, _ = extract_championship_features(X_test)
    
    # Ensure we have the same columns
    missing_cols = set(feature_columns) - set(X_test_features.columns)
    for col in missing_cols:
        X_test_features[col] = 0
    
    # Reorder columns to match training
    X_test_features = X_test_features[feature_columns]
    
    # Combine features
    X_test_combined = hstack([X_test_text, X_test_features.values])
    
    # Make predictions
    predictions = model.predict(X_test_combined)
    
    return predictions

def main():
    """
    Main championship pipeline
    """
    print("🏆 CHAMPIONSHIP SOLUTION FOR NLP DISASTER TWEETS")
    print("=" * 60)
    
    # File paths
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    submission_path = "championship_submission.csv"
    
    try:
        # Load data
        print("📥 Loading competition data...")
        train_df, test_df = load_data(train_path, test_path)
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Test samples: {len(test_df):,}")
        print(f"   Disaster rate in train: {train_df['target'].mean():.1%}")
        
        # Prepare data
        print("\n📋 Preparing features...")
        X_train = train_df['text'].values
        y_train = train_df['target'].values
        X_test = test_df['text'].values
        test_ids = test_df['id'].values
        
        # Create and train championship model
        print("\n🤖 Building championship model...")
        model, vectorizer, feature_columns = create_championship_ensemble(X_train, y_train)
        
        # Generate predictions
        print("\n🎯 Making predictions...")
        predictions = predict_championship(model, vectorizer, feature_columns, X_test)
        
        # Create submission
        print("\n📝 Creating submission file...")
        submission_df = pd.DataFrame({
            'id': test_ids,
            'target': predictions.astype(int)
        })
        submission_df.to_csv(submission_path, index=False)
        
        # Results
        disaster_count = predictions.sum()
        total_count = len(predictions)
        disaster_rate = disaster_count / total_count
        
        print("=" * 60)
        print("🏆 CHAMPIONSHIP SOLUTION COMPLETE!")
        print(f"   📊 Predictions: {total_count:,}")
        print(f"   🚨 Disaster predictions: {disaster_count:,} ({disaster_rate:.1%})")
        print(f"   ✅ Safe predictions: {total_count - disaster_count:,} ({(1-disaster_rate):.1%})")
        print(f"   💾 Submission saved to: {submission_path}")
        print("=" * 60)
        print("💡 Tips for even better performance:")
        print("   - Consider transformer models (BERT/Roberta) with fine-tuning")
        print("   - Use domain-specific word embeddings for disaster tweets")
        print("   - Apply test-time augmentation (TTA)")
        print("   - Experiment with different disaster keyword weights")
        print("   - Try linguistic feature selection with mutual information")
        print("🍀 Good luck - go get that 1st place! 🥇")
        
    except Exception as e:
        print(f"❌ Error in championship solution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()