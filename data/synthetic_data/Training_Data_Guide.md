# Synthetic Training Data - Complete Guide & Usage Examples

## Overview

You now have **14 comprehensive synthetic training datasets** totaling **3,700+ training samples** across all key ML solution areas for the AdvisoryAI Agentic Chaser challenge. All data is in both JSON and CSV formats for maximum flexibility.

---

## Dataset Inventory

### Core Operational Datasets (JSON + CSV)
| # | Dataset | Samples | Use Case |
|---|---------|---------|----------|
| 1 | Client Profiles | 50 | CRM data, demographics, engagement patterns |
| 2 | Historical LOAs | 100 | Workflow state tracking, SLA monitoring |
| 3 | Chatbot Conversations | 300 | Training data for document request conversations |
| 4 | Document Submissions | 1,000 | OCR results, document quality, validation |
| 5 | Post-Advice Items | 200 | Task tracking, completion prediction |
| 6 | Communication Log | 500 | Sentiment analysis, intent detection |
| 7 | Case Workflows | 150 | Case state management, timeline tracking |
| 8 | Provider Response Times | 400 | Historical response patterns, forecasting |

### ML-Ready Training Datasets (CSV)
| # | Dataset | Samples | Algorithm Purpose |
|---|---------|---------|-------------------|
| 9 | OCR Training Data | 1,000 | Document classification, confidence scoring |
| 10 | Sentiment Training Data | 500 | Sentiment analysis, emotion classification |
| 11 | Priority Scoring Data | 100 | Regression model for priority calculation |
| 12 | Time Series Provider Data | 400 | ARIMA/Prophet for response time forecasting |
| 13 | Chatbot Intent Training | 300 | NLP intent classification, entity extraction |
| 14 | Engagement Prediction Data | 200 | Binary classification for completion likelihood |

---

## Solution-Specific Training Guides

### SOLUTION 1: Document Processing (OCR + ML + NLP)

**Relevant Datasets:**
- `04_document_submissions.csv` - Document metadata and OCR results
- `09_ocr_training_data.csv` - Pre-processed for ML training

**Model 1: Document Type Classification**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('04_document_submissions.csv')

# Prepare features
le_subtype = LabelEncoder()
X = pd.DataFrame({
    'quality_score': df['document_quality_score'],
    'has_issues': (df['quality_issues'].str.len() > 2).astype(int),
    'upload_month': pd.to_datetime(df['upload_date']).dt.month
})

y = df['document_type']

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

print(f"Document Classification Accuracy: {clf.score(X, y):.2%}")
```

**Model 2: OCR Confidence Prediction**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Prepare regression target
df_ocr = pd.read_csv('09_ocr_training_data.csv')

X = df_ocr[['has_quality_issues', 'quality_issue_count', 'upload_month']]
y = df_ocr['ocr_confidence_score']

# Train regressor
reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg.fit(X, y)

print(f"OCR Confidence R² Score: {reg.score(X, y):.3f}")
```

**Model 3: Document Validation Rules**
```python
# Simple rule-based validation
def validate_document(doc_type, quality_score, has_issues):
    # If quality issues exist
    if has_issues:
        return False
    
    # If OCR confidence too low
    if quality_score < 85:
        return False
    
    # Type-specific validation
    if doc_type in ['Passport', 'Driving Licence']:
        return quality_score > 90  # Higher bar for identity docs
    
    return quality_score > 80

# Apply to data
df['validation_passed_ml'] = df.apply(
    lambda row: validate_document(
        row['document_type'], 
        row['ocr_confidence_score'],
        len(eval(row['quality_issues'])) > 0
    ), axis=1
)

# Compare with actual
accuracy = (df['validation_passed_ml'] == df['validation_passed']).mean()
print(f"Validation Rule Accuracy: {accuracy:.2%}")
```

---

### SOLUTION 2: Chatbot (NLP + Intent Recognition)

**Relevant Datasets:**
- `03_chatbot_training_conversations.csv` - Full conversations
- `13_chatbot_intent_training.csv` - Pre-processed for intent classification

**Model 1: Intent Classification**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv('13_chatbot_intent_training.csv')

# Create pipeline
clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),
    ('nb', MultinomialNB())
])

# Train
X = df['client_message']
y = df['intent']
clf.fit(X, y)

print(f"Intent Classification Accuracy: {clf.score(X, y):.2%}")

# Predict on new message
test_message = "I don't have a recent payslip"
intent = clf.predict([test_message])[0]
print(f"Predicted Intent: {intent}")
```

**Model 2: Document Type Extraction (NER-style)**
```python
# Extract document mentioned in conversation
def extract_document_mention(message):
    document_keywords = {
        'Passport': ['passport', 'travel'],
        'Driving Licence': ['driving', 'licence', 'license'],
        'Payslip': ['payslip', 'salary', 'wage'],
        'P60': ['p60', 'tax year'],
        'Utility Bill': ['utility', 'bill', 'electricity', 'water'],
        'Bank Statement': ['bank', 'statement'],
    }
    
    message_lower = message.lower()
    for doc_type, keywords in document_keywords.items():
        for keyword in keywords:
            if keyword in message_lower:
                return doc_type
    return None

# Test extraction
test_msg = "I found my P60 from 2023"
doc = extract_document_mention(test_msg)
print(f"Extracted Document Type: {doc}")
```

**Model 3: Response Generation (Template-based)**
```python
response_templates = {
    'Passport': "Great! Your passport works perfectly for ID verification. It expires in [X] years.",
    'Payslip': "Got it! If you don't have a recent payslip, a P60 from the last tax year works too.",
    'Utility Bill': "Perfect! Your utility bill is within 3 months - that's exactly what we need for address proof.",
    'P60': "Thanks for the P60! If it's from the last tax year, that's fine. Otherwise we need the most recent one."
}

# Select response based on detected intent
def generate_response(message):
    doc_type = extract_document_mention(message)
    if doc_type in response_templates:
        return response_templates[doc_type]
    return "Thanks for your message. Could you clarify which document you're referring to?"

print(generate_response("I've attached my P60 from 2024"))
```

---

### SOLUTION 3: Provider Communication & State Management

**Relevant Datasets:**
- `02_historical_loas.csv` - LOA state tracking
- `08_provider_response_times.csv` - Provider timing patterns
- `12_time_series_provider_data.csv` - For forecasting

**Model 1: Provider Response Time Forecasting**
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_csv('12_time_series_provider_data.csv')

# Group by provider and calculate statistics
provider_stats = df.groupby('provider').agg({
    'days_to_response': ['mean', 'std', 'min', 'max'],
    'sla_compliant': 'mean'
}).round(2)

print("Provider Response Time Statistics:")
print(provider_stats)

# Train predictor
df_encoded = df.copy()
df_encoded['provider_encoded'] = pd.factorize(df['provider'])[0]
df_encoded['is_q4'] = (df_encoded['quarter'] == 'Q4').astype(int)

X = df_encoded[['provider_encoded', 'month', 'is_q4', 'response_completeness_score']]
y = df_encoded['days_to_response']

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

print(f"\nResponse Time Prediction R² Score: {model.score(X, y):.3f}")

# Predict for new submission
prediction = model.predict([[1, 11, 0, 0.9]])[0]
print(f"Predicted Response Time: {prediction:.0f} days")
```

**Model 2: SLA Compliance Prediction**
```python
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('12_time_series_provider_data.csv')

# Features
X = df[['month']].copy()
X['provider_encoded'] = pd.factorize(df['provider'])[0]

# Target: whether SLA was met
y = df['sla_compliant'].astype(int)

clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

print(f"SLA Compliance Prediction Accuracy: {clf.score(X, y):.2%}")

# Feature importance
importance = pd.DataFrame({
    'feature': ['month', 'provider'],
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)
```

**Model 3: State Transition Logic**
```python
# Define valid state transitions
state_transitions = {
    'Prepared': ['Client Signature Pending'],
    'Client Signature Pending': ['Client Signed'],
    'Client Signed': ['Provider Submitted'],
    'Provider Submitted': ['Provider Processing'],
    'Provider Processing': ['Information Received', 'Provider Processing'],  # Can stay or progress
    'Information Received': ['Complete'],
    'Complete': ['Complete']  # Terminal state
}

def suggest_next_action(current_state, days_in_state, provider_sla):
    """Suggest action based on state and time elapsed"""
    
    actions = {
        'Client Signature Pending': 'Send SMS reminder to client' if days_in_state > 5 else 'Wait for signature',
        'Provider Submitted': 'Track status' if days_in_state < provider_sla else 'Call provider to check',
        'Provider Processing': 'Follow up with provider' if days_in_state > provider_sla else f'Expected completion in {provider_sla - days_in_state} days',
        'Information Received': 'Review information and prepare next steps',
        'Complete': 'Archive case'
    }
    
    return actions.get(current_state, 'Review case status')

# Test
print(suggest_next_action('Provider Submitted', 8, 15))
print(suggest_next_action('Provider Processing', 16, 15))
```

---

### SOLUTION 4: Sentiment Analysis & Engagement

**Relevant Datasets:**
- `06_communication_sentiment_log.csv` - Communication logs
- `10_sentiment_training_data.csv` - Pre-processed for sentiment

**Model 1: Sentiment Classifier**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv('10_sentiment_training_data.csv')

# Create classification pipeline
clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
    ('lr', LogisticRegression(max_iter=200, random_state=42))
])

# Train
X = df['message_text']
y = df['sentiment_label']
clf.fit(X, y)

print(f"Sentiment Classification Accuracy: {clf.score(X, y):.2%}")

# Test on new messages
test_messages = [
    "Thanks so much for helping!",
    "This is taking forever",
    "I'm not sure what you mean"
]

for msg in test_messages:
    sentiment = clf.predict([msg])[0]
    confidence = max(clf.predict_proba([msg])[0])
    print(f"'{msg}' → {sentiment} ({confidence:.0%})")
```

**Model 2: Frustration Detection**
```python
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('10_sentiment_training_data.csv')

# Features
df['has_urgency'] = df['contains_urgency_words'].astype(int)
df['has_frustration'] = df['contains_frustration_words'].astype(int)
df['is_frustrated'] = (df['sentiment_label'] == 'Frustrated').astype(int)

X = df[['message_length_words', 'has_urgency', 'has_frustration']]
y = df['is_frustrated']

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

print(f"Frustration Detection Accuracy: {clf.score(X, y):.2%}")

# Feature importance
print("\nFrustration Indicators:")
for feature, importance in zip(X.columns, clf.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
```

**Model 3: Escalation Rules**
```python
def should_escalate_to_human(message, sentiment, days_outstanding):
    """Determine if message should be escalated to human advisor"""
    
    escalation_score = 0
    
    # High frustration
    if sentiment == 'Frustrated':
        escalation_score += 50
    
    # Confusion
    if sentiment == 'Confused':
        escalation_score += 30
    
    # Long outstanding
    if days_outstanding > 10:
        escalation_score += 20
    
    # Question mark (uncertainty)
    if '?' in message:
        escalation_score += 15
    
    return escalation_score > 50

# Test
test_cases = [
    ("Thanks!", "Positive", 2),
    ("Still waiting!", "Frustrated", 12),
    ("I don't understand", "Confused", 5),
    ("This is ridiculous", "Frustrated", 15),
]

for msg, sentiment, days in test_cases:
    escalate = should_escalate_to_human(msg, sentiment, days)
    print(f"Message: '{msg}' | Escalate: {escalate}")
```

---

### SOLUTION 5: Priority Scoring & Workflow

**Relevant Datasets:**
- `11_priority_scoring_data.csv` - Priority calculation training
- `14_engagement_prediction_data.csv` - Completion prediction

**Model 1: Priority Scoring**
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('11_priority_scoring_data.csv')

# Features for priority calculation
X = df[[
    'days_in_current_state',
    'sla_overdue',
    'document_quality_score',
    'client_age_55_plus'
]]

y = df['priority_score_calculated']

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print(f"Priority Score Prediction R² Score: {model.score(X, y):.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nPriority Score Drivers:")
print(importance)

# Predict priority for new case
test_case = [[18, 3, 75, 1]]  # 18 days elapsed, 3 days overdue, 75% quality, age 55+
predicted_priority = model.predict(test_case)[0]
print(f"\nExample Case Priority Score: {predicted_priority:.2f}")
```

**Model 2: Completion Prediction**
```python
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('14_engagement_prediction_data.csv')

# Features
X = df[[
    'days_outstanding',
    'opened',
    'completion_percentage',
    'days_since_last_interaction'
]]

y = df['likely_to_complete'].astype(int)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

print(f"Completion Prediction Accuracy: {clf.score(X, y):.2%}")

# Test on new item
# Item sent 7 days ago, was opened, 25% complete, last interaction 2 days ago
test_item = [[7, 1, 25, 2]]
completion_prob = clf.predict_proba(test_item)[0][1]
print(f"\nItem Completion Probability: {completion_prob:.0%}")
```

**Model 3: Daily Priority Dashboard**
```python
def generate_daily_priorities(cases_df, max_items=5):
    """Generate priority-ranked list for advisor's day"""
    
    # Calculate priority score
    cases_df['priority_score'] = (
        (cases_df['days_in_current_state'] * 0.3) +
        (cases_df['sla_overdue'].fillna(0) * 0.4) +
        (cases_df['client_age_55_plus'] * 0.2) +
        (cases_df['document_quality_score'] / 100 * 0.1)
    )
    
    # Rank by priority
    priority_cases = cases_df.nlargest(max_items, 'priority_score')[
        ['loa_id', 'provider', 'priority_level', 'priority_score', 'days_in_current_state']
    ]
    
    return priority_cases

# Example
df = pd.read_csv('11_priority_scoring_data.csv')
daily_priorities = generate_daily_priorities(df)
print("Today's Priority Items:")
print(daily_priorities)
```

---

## Quick Start: Running All Models

```python
# Load all datasets
import pandas as pd

datasets = {
    'clients': pd.read_csv('01_client_profiles.csv'),
    'loas': pd.read_csv('02_historical_loas.csv'),
    'conversations': pd.read_csv('03_chatbot_training_conversations.csv'),
    'documents': pd.read_csv('04_document_submissions.csv'),
    'post_advice': pd.read_csv('05_post_advice_items.csv'),
    'communications': pd.read_csv('06_communication_sentiment_log.csv'),
    'workflows': pd.read_csv('07_case_workflow_states.csv'),
    'provider_times': pd.read_csv('08_provider_response_times.csv'),
}

# Explore
for name, df in datasets.items():
    print(f"\n{name}: {len(df)} rows, {len(df.columns)} columns")
    print(df.dtypes)
```

---

## Data Quality & Statistics

### Client Distribution
- Age Range: 22-75 years (mean: 48, median: 47)
- Employment: 20% Employed, 20% Self-Employed, 20% Retired, 20% Student, 20% Unemployed
- Risk Profiles: Distributed across Conservative-Aggressive spectrum
- Engagement: 25% Very High, 25% High, 30% Medium, 20% Low

### LOA Processing
- Average Days in Processing: 12-18 days (varies by provider)
- SLA Compliance Rate: 65-75% of LOAs meet SLA
- Document Quality: Mean 72/100, bimodal distribution (good or problematic)
- Signature Verification: 80% correctly signed

### Document Quality
- OCR Confidence: Mean 92%, std dev 8%
- Quality Issues Present: 30% of documents
- Validation Pass Rate: 70% on first submission

### Communication Sentiment
- Positive: 40% of messages
- Neutral: 30% of messages
- Frustrated: 20% of messages
- Confused: 10% of messages

---

## Next Steps

1. **Fork/Clone** this synthetic data
2. **Train models** using provided examples
3. **Evaluate** model performance on your validation split
4. **Integrate** trained models into your architecture
5. **Iterate** on feature engineering and hyperparameters
6. **Deploy** to production with confidence

Good luck with your solution!
