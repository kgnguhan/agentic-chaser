# Synthetic Training Data for AdvisoryAI Agentic Chaser

## Overview
This directory contains comprehensive synthetic training datasets for building ML/AI solutions for the AdvisoryAI Hack-to-Hire challenge. All data is realistic but completely synthetic and suitable for immediate model training.

## Datasets

### Core Datasets (JSON + CSV)
1. **01_client_profiles.json/csv** - 50 UK IFA client profiles with demographics and preferences
2. **02_historical_loas.json/csv** - 100 Letter of Authority records at various processing stages
3. **03_chatbot_training_conversations.json/csv** - 300 document request conversations
4. **04_document_submissions.json/csv** - 1,000 document submission records with OCR results
5. **05_post_advice_items.json/csv** - 200 post-advice task tracking items
6. **06_communication_sentiment_log.json/csv** - 500 client communication messages with sentiment
7. **07_case_workflow_states.json/csv** - 150 case workflow tracking records
8. **08_provider_response_times.json/csv** - 400 historical provider response time records

### ML-Specific Training Datasets (CSV)
9. **09_ocr_training_data.csv** - Document classification training (1,000 samples)
10. **10_sentiment_training_data.csv** - Sentiment analysis training (500 samples)
11. **11_priority_scoring_data.csv** - Priority calculation training (100 samples)
12. **12_time_series_provider_data.csv** - Time series forecasting (400 samples)
13. **13_chatbot_intent_training.csv** - Intent classification training (300 samples)
14. **14_engagement_prediction_data.csv** - Completion prediction training (200 samples)

## Use Cases

### Solution 1: Document Processing (OCR + ML)
- Use: `04_document_submissions.csv` and `09_ocr_training_data.csv`
- Train document classifier, OCR confidence predictor, quality assessment model

### Solution 2: Chatbot (NLP + Intent Recognition)
- Use: `03_chatbot_training_conversations.csv` and `13_chatbot_intent_training.csv`
- Fine-tune LLM on document request conversations
- Train intent classifier

### Solution 3: Provider Communication & State Management
- Use: `02_historical_loas.csv`, `08_provider_response_times.csv`, `12_time_series_provider_data.csv`
- Build time series forecasting model for provider response times
- Train state transition logic

### Solution 4: Sentiment Analysis & Engagement
- Use: `06_communication_sentiment_log.csv` and `10_sentiment_training_data.csv`
- Train sentiment classifier
- Build engagement/frustration detection model

### Solution 5: Priority Scoring & Workflow
- Use: `11_priority_scoring_data.csv` and `14_engagement_prediction_data.csv`
- Train priority prediction model
- Build completion probability classifier

## Data Characteristics

### Client Demographics
- Age: 22-75 years
- Employment: Employed, Self-Employed, Retired, Student, Unemployed
- Annual Income: £0-£250k depending on employment type
- Risk Profiles: Conservative, Moderate, Balanced, Growth, Aggressive
- Document Responsiveness: Very High, High, Medium, Low

### LOA Processing
- Providers: 10 major UK pension providers (Aviva, Standard Life, Fidelity, etc.)
- States: 7 workflow states from Prepared to Complete
- SLA: 10-20 days depending on provider
- Quality Scores: 0-100 scale for document completeness

### Documents
- Types: Passport, Driving Licence, Utility Bill, P60, Payslip, Bank Statement, etc.
- OCR Confidence: 50-100 scale, normally distributed around 92%
- Quality Issues: Blurry, Damaged, Partial, Wrong Document, Expired

### Communication
- Channels: Email, SMS, WhatsApp, Phone
- Sentiments: Positive (40%), Neutral (30%), Frustrated (20%), Confused (10%)
- Message Length: 5-30 words

## Quick Start

```python
import pandas as pd

# Load data
clients = pd.read_csv('01_client_profiles.csv')
loas = pd.read_csv('02_historical_loas.csv')
documents = pd.read_csv('04_document_submissions.csv')

# Explore
print(clients.head())
print(loas['current_state'].value_counts())
print(documents['ocr_confidence_score'].describe())
```

## Key Statistics

| Dataset | Samples | Key Metric |
|---------|---------|-----------|
| Clients | 50 | Age: 22-75 |
| LOAs | 100 | SLA Overdue: 35% |
| Conversations | 300 | Correct Response: 80% |
| Documents | 1,000 | OCR Confidence: 92% avg |
| Post-Advice Items | 200 | Completion Rate: 30% |
| Messages | 500 | Positive Sentiment: 40% |
| Case Workflows | 150 | Avg Days in Progress: 45 |
| Provider Data | 400 | SLA Compliant: 75% |

## Notes

- All data is synthetic and realistic
- No real personal information included
- Suitable for immediate model training
- Data represents typical UK IFA workflow patterns
- Time-based features use February 1, 2026 as reference

## License

These datasets are provided for the AdvisoryAI Hack-to-Hire challenge only.
