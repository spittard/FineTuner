# Proposal 2: Algorithm & Accuracy Improvements

**Document Version:** 1.0
**Created:** 2026-04-22
**Focus Area:** Matching Quality, Learning from Feedback, Domain Adaptation

---

## Executive Summary

This proposal focuses on improving match accuracy through algorithm enhancements, machine learning integration, and feedback-driven optimization. The goal is to increase precision on edge cases while maintaining high recall.

**Key Outcomes:**
- 5-10% improvement in control set accuracy
- Learning from user corrections
- Domain-specific weight tuning
- Better handling of abbreviations, variants, and misspellings

---

## 1. Learnable Score Weights

### 1.1 Problem

Fixed weights (50/25/25 for string/semantic/concept) may not be optimal for all query types or domains. Some queries benefit more from lexical matching, others from semantic.

### 1.2 Solution: Query-Adaptive Weights

```python
class AdaptiveWeightModel:
    """Learn optimal weights based on query characteristics."""

    def __init__(self):
        # Base weights (current defaults)
        self.base_weights = {
            'string': 0.50,
            'semantic': 0.25,
            'concept': 0.25
        }

        # Query feature extractors
        self.features = [
            'query_length',           # Number of words
            'has_acronym',            # Query looks like acronym
            'has_numbers',            # Contains numbers (addresses, etc.)
            'avg_word_length',        # Long words = more specific
            'stop_word_ratio',        # Generic vs specific
            'has_location_words',     # Geographic terms
            'has_industry_terms',     # Industry-specific terms
        ]

        # Learned adjustments (initialize to zero)
        self.weight_adjustments = np.zeros((len(self.features), 3))

    def extract_features(self, query: str) -> np.ndarray:
        """Extract features from query."""
        words = query.lower().split()
        clean_words = [w for w in words if w not in STOP_WORDS]

        features = np.array([
            len(words),                                      # query_length
            1.0 if len(query) < 6 and query.isupper() else 0,  # has_acronym
            1.0 if any(c.isdigit() for c in query) else 0,   # has_numbers
            np.mean([len(w) for w in clean_words]) if clean_words else 0,  # avg_word_length
            1 - len(clean_words) / len(words) if words else 0,  # stop_word_ratio
            1.0 if any(w in LOCATION_WORDS for w in words) else 0,  # has_location
            1.0 if any(w in INDUSTRY_WORDS for w in words) else 0,  # has_industry
        ])

        return features

    def get_weights(self, query: str) -> Dict[str, float]:
        """Get adaptive weights for query."""
        features = self.extract_features(query)

        # Compute adjustments
        adjustments = features @ self.weight_adjustments

        # Apply adjustments and normalize
        weights = np.array([
            self.base_weights['string'] + adjustments[0],
            self.base_weights['semantic'] + adjustments[1],
            self.base_weights['concept'] + adjustments[2]
        ])

        # Ensure valid weights (positive, sum to 1)
        weights = np.maximum(weights, 0.05)
        weights = weights / weights.sum()

        return {
            'string': weights[0],
            'semantic': weights[1],
            'concept': weights[2]
        }

    def train(self, training_data: List[Dict]):
        """
        Train weight adjustments from labeled data.

        training_data format:
        [
            {
                'query': 'IBM',
                'correct_match': 'International Business Machines',
                'incorrect_matches': ['IBM Consulting', 'IBMS Inc']
            },
            ...
        ]
        """
        from sklearn.linear_model import LogisticRegression

        # Build training examples
        X, y = [], []

        for example in training_data:
            query = example['query']
            features = self.extract_features(query)

            # Positive example
            X.append(features)
            y.append(1)

            # Negative examples (from incorrect matches)
            for _ in example.get('incorrect_matches', []):
                X.append(features)
                y.append(0)

        # Train model to predict correct weight adjustments
        # (Simplified - real implementation would be more sophisticated)
        model = LogisticRegression()
        model.fit(X, y)

        # Extract learned adjustments
        self.weight_adjustments = model.coef_.reshape(-1, 3)
```

### 1.3 Integration

```python
class CompanyMatcher:
    def __init__(self, ...):
        ...
        self.adaptive_weights = AdaptiveWeightModel()

    def match_with_location(self, query, ...):
        # Get adaptive weights
        weights = self.adaptive_weights.get_weights(query)

        # Use in scoring
        base_score = (
            string_score * weights['string'] +
            sem_score_norm * weights['semantic'] +
            concept_alignment * weights['concept']
        )
```

---

## 2. Feedback Loop System

### 2.1 Problem

No mechanism to learn from user corrections or track match quality over time.

### 2.2 Solution: Feedback Collection & Training

```python
# feedback_service.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json

@dataclass
class FeedbackEntry:
    """User feedback on a match."""
    timestamp: datetime
    query: str
    query_city: Optional[str]
    query_state: Optional[str]
    shown_match: str
    correct_match: Optional[str]  # None if shown_match was correct
    feedback_type: str  # 'accept', 'reject', 'correct'
    user_id: Optional[str]

class FeedbackService:
    """Collect and analyze user feedback."""

    def __init__(self, storage_path: str = 'feedback_data.jsonl'):
        self.storage_path = storage_path
        self.feedback: List[FeedbackEntry] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing feedback from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    self.feedback.append(FeedbackEntry(**data))
        except FileNotFoundError:
            pass

    def record(self, entry: FeedbackEntry):
        """Record new feedback entry."""
        self.feedback.append(entry)

        # Append to storage
        with open(self.storage_path, 'a') as f:
            data = {
                'timestamp': entry.timestamp.isoformat(),
                'query': entry.query,
                'query_city': entry.query_city,
                'query_state': entry.query_state,
                'shown_match': entry.shown_match,
                'correct_match': entry.correct_match,
                'feedback_type': entry.feedback_type,
                'user_id': entry.user_id
            }
            f.write(json.dumps(data) + '\n')

    def get_training_data(self) -> List[Dict]:
        """Convert feedback to training data for weight learning."""
        training_data = []

        for entry in self.feedback:
            if entry.feedback_type == 'accept':
                training_data.append({
                    'query': entry.query,
                    'correct_match': entry.shown_match,
                    'incorrect_matches': []
                })
            elif entry.feedback_type == 'correct' and entry.correct_match:
                training_data.append({
                    'query': entry.query,
                    'correct_match': entry.correct_match,
                    'incorrect_matches': [entry.shown_match]
                })
            elif entry.feedback_type == 'reject':
                training_data.append({
                    'query': entry.query,
                    'correct_match': None,  # Unknown
                    'incorrect_matches': [entry.shown_match]
                })

        return training_data

    def get_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics from feedback."""
        total = len(self.feedback)
        if total == 0:
            return {'accuracy': None, 'total': 0}

        correct = sum(1 for f in self.feedback if f.feedback_type == 'accept')
        corrected = sum(1 for f in self.feedback if f.feedback_type == 'correct')
        rejected = sum(1 for f in self.feedback if f.feedback_type == 'reject')

        return {
            'accuracy': correct / total,
            'correction_rate': corrected / total,
            'rejection_rate': rejected / total,
            'total': total,
            'by_day': self._accuracy_by_day()
        }

    def _accuracy_by_day(self) -> Dict[str, float]:
        """Calculate accuracy trends by day."""
        from collections import defaultdict

        by_day = defaultdict(lambda: {'correct': 0, 'total': 0})

        for entry in self.feedback:
            day = entry.timestamp.strftime('%Y-%m-%d')
            by_day[day]['total'] += 1
            if entry.feedback_type == 'accept':
                by_day[day]['correct'] += 1

        return {
            day: data['correct'] / data['total']
            for day, data in by_day.items()
        }
```

### 2.3 API Endpoints

```python
# app.py additions

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on a match result."""
    data = request.get_json()

    entry = FeedbackEntry(
        timestamp=datetime.now(),
        query=data['query'],
        query_city=data.get('city'),
        query_state=data.get('state'),
        shown_match=data['shown_match'],
        correct_match=data.get('correct_match'),
        feedback_type=data['feedback_type'],
        user_id=data.get('user_id')
    )

    feedback_service.record(entry)

    return jsonify({'success': True})

@app.route('/api/feedback/metrics', methods=['GET'])
def get_feedback_metrics():
    """Get accuracy metrics from feedback."""
    return jsonify(feedback_service.get_accuracy_metrics())

@app.route('/api/feedback/retrain', methods=['POST'])
def retrain_weights():
    """Retrain adaptive weights from feedback."""
    training_data = feedback_service.get_training_data()

    if len(training_data) < 100:
        return jsonify({'error': 'Need at least 100 feedback entries'}), 400

    matcher.adaptive_weights.train(training_data)

    return jsonify({
        'success': True,
        'training_examples': len(training_data)
    })
```

---

## 3. Enhanced Acronym Handling

### 3.1 Problem

Current acronym handling:
- Ignores 2-letter acronyms (misses NY, DC, CA)
- Doesn't handle lowercase acronyms
- Misses industry-standard abbreviations (LLC, PA, etc.)

### 3.2 Solution: Multi-Strategy Acronym Engine

```python
class AcronymEngine:
    """Enhanced acronym detection and expansion."""

    # Known abbreviations that should be expanded
    KNOWN_ABBREVIATIONS = {
        # Corporate suffixes
        'llc': 'limited liability company',
        'inc': 'incorporated',
        'corp': 'corporation',
        'ltd': 'limited',
        'pc': 'professional corporation',
        'pa': 'professional association',
        'plc': 'public limited company',

        # Industry terms
        'hq': 'headquarters',
        'intl': 'international',
        'natl': 'national',
        'svcs': 'services',
        'assoc': 'association',
        'mgmt': 'management',

        # Geographic (2-letter state codes handled separately)
        'nyc': 'new york city',
        'sf': 'san francisco',
        'la': 'los angeles',
        'dc': 'district of columbia',
    }

    # Valid 2-letter acronyms that are NOT state codes
    VALID_TWO_LETTER = {
        'ge', 'gm', 'at', 'hp', 'bp', 'ey', 'pw', 'ub', 'jp',
        'pg', 'jc', '3m', 'td', 'bb', 'bt', 'ups', 'ibm'  # Extended
    }

    @classmethod
    def is_acronym(cls, text: str) -> bool:
        """Determine if text is likely an acronym."""
        text = text.strip()

        # All uppercase letters
        if text.isupper() and len(text) >= 2:
            return True

        # Known abbreviation
        if text.lower() in cls.KNOWN_ABBREVIATIONS:
            return True

        # Letters with periods (I.B.M., A.B.A.)
        if re.match(r'^[A-Z](\.[A-Z])+\.?$', text):
            return True

        # CamelCase acronym in name (IBM, ABA)
        if re.match(r'^[A-Z]{2,5}$', text):
            return True

        return False

    @classmethod
    def expand_acronym(cls, acronym: str) -> List[str]:
        """Get possible expansions for an acronym."""
        expansions = []
        acr_lower = acronym.lower()

        # Check known abbreviations
        if acr_lower in cls.KNOWN_ABBREVIATIONS:
            expansions.append(cls.KNOWN_ABBREVIATIONS[acr_lower])

        return expansions

    @classmethod
    def should_process_two_letter(cls, text: str) -> bool:
        """Check if 2-letter text should be processed as acronym."""
        text_lower = text.lower()

        # Not a state code
        if text_lower in TextPreprocessor.STATE_ABBREV:
            return False

        # Is a known valid 2-letter company
        if text_lower in cls.VALID_TWO_LETTER:
            return True

        # Context-dependent: check if surrounded by company context
        # (Would need full query context)

        return False

    @classmethod
    def generate_variants(cls, name: str) -> List[str]:
        """Generate acronym and abbreviation variants of a name."""
        variants = []

        # Standard acronym
        acronym = TextPreprocessor.generate_acronym(name)
        if acronym:
            variants.append(acronym)
            variants.append(acronym.lower())

        # With periods
        if acronym and len(acronym) <= 5:
            variants.append('.'.join(acronym) + '.')

        # Without vowels (common abbreviation pattern)
        words = name.split()
        if len(words) == 1 and len(name) > 4:
            no_vowels = ''.join(c for c in name.lower() if c not in 'aeiou')
            if len(no_vowels) >= 3:
                variants.append(no_vowels)

        return variants
```

### 3.3 Integration

```python
class CompanyMatcher:
    def match_with_location(self, query, ...):
        # Phase 0: Enhanced acronym handling
        if AcronymEngine.is_acronym(query):
            # Include 2-letter if valid
            if len(query) == 2 and AcronymEngine.should_process_two_letter(query):
                self._process_acronym_expansion(query, candidates)
            elif len(query) > 2:
                self._process_acronym_expansion(query, candidates)
```

---

## 4. Typo & Misspelling Tolerance

### 4.1 Problem

No tolerance for common misspellings or typos in queries.

### 4.2 Solution: Fuzzy Query Preprocessing

```python
class FuzzyQueryProcessor:
    """Handle typos and misspellings in queries."""

    def __init__(self, company_names: List[str]):
        # Build phonetic index for fuzzy matching
        self.soundex_index = self._build_soundex_index(company_names)
        self.bigram_index = self._build_bigram_index(company_names)

    def _build_soundex_index(self, names: List[str]) -> Dict[str, List[int]]:
        """Build soundex-based index for phonetic matching."""
        index = {}
        for i, name in enumerate(names):
            for word in name.split():
                soundex = self._soundex(word)
                if soundex not in index:
                    index[soundex] = []
                index[soundex].append(i)
        return index

    def _build_bigram_index(self, names: List[str]) -> Dict[str, Set[int]]:
        """Build character bigram index."""
        index = {}
        for i, name in enumerate(names):
            name_lower = name.lower()
            for j in range(len(name_lower) - 1):
                bigram = name_lower[j:j+2]
                if bigram not in index:
                    index[bigram] = set()
                index[bigram].add(i)
        return index

    def _soundex(self, word: str) -> str:
        """Generate Soundex code."""
        # Standard Soundex implementation
        word = word.upper()
        if not word:
            return "0000"

        codes = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        result = word[0]
        for char in word[1:]:
            code = codes.get(char, '0')
            if code != '0' and code != result[-1]:
                result += code

        return (result + '0000')[:4]

    def expand_query(self, query: str, max_variants: int = 5) -> List[str]:
        """Generate variant spellings for fuzzy matching."""
        variants = [query]

        # Common typo patterns
        typo_variants = self._generate_typo_variants(query)
        variants.extend(typo_variants[:max_variants])

        # Phonetic matches from index
        phonetic_matches = self._get_phonetic_matches(query)
        variants.extend(phonetic_matches[:max_variants])

        return list(set(variants))

    def _generate_typo_variants(self, query: str) -> List[str]:
        """Generate common typo variants."""
        variants = []

        # Character swaps
        for i in range(len(query) - 1):
            swapped = query[:i] + query[i+1] + query[i] + query[i+2:]
            variants.append(swapped)

        # Common substitutions
        subs = {
            'ie': 'ei', 'ei': 'ie',  # i before e
            'ph': 'f', 'f': 'ph',     # phone/fone
            'ck': 'k', 'k': 'ck',     # check/chek
            'tion': 'sion', 'sion': 'tion',  # action/acsion
        }

        for old, new in subs.items():
            if old in query.lower():
                variants.append(query.lower().replace(old, new))

        return variants

    def _get_phonetic_matches(self, query: str) -> List[str]:
        """Get phonetically similar company names."""
        matches = set()

        for word in query.split():
            soundex = self._soundex(word)
            if soundex in self.soundex_index:
                for idx in self.soundex_index[soundex][:10]:
                    matches.add(idx)

        return list(matches)

    def get_fuzzy_candidates(self, query: str, k: int = 100) -> List[int]:
        """Get candidate indices using fuzzy matching."""
        candidates = set()

        # Bigram similarity
        query_bigrams = set(query[i:i+2] for i in range(len(query)-1))

        for bigram in query_bigrams:
            if bigram in self.bigram_index:
                candidates.update(self.bigram_index[bigram])

        # Score and rank by bigram overlap
        scored = []
        for idx in candidates:
            name_bigrams = set()
            for i in range(len(self.names[idx]) - 1):
                name_bigrams.add(self.names[idx][i:i+2].lower())

            overlap = len(query_bigrams & name_bigrams) / len(query_bigrams | name_bigrams)
            scored.append((idx, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored[:k]]
```

---

## 5. Domain-Specific Models

### 5.1 Problem

Single embedding model may not capture domain-specific nuances (legal, medical, tech).

### 5.2 Solution: Ensemble of Domain Models

```python
class DomainEnsemble:
    """Ensemble of domain-specific embedding models."""

    def __init__(self):
        self.models = {
            'general': SentenceTransformer('all-MiniLM-L6-v2'),
            'legal': SentenceTransformer('legal-bert-base-uncased'),  # Example
            'medical': SentenceTransformer('pubmedbert-base'),        # Example
        }

        self.domain_classifier = self._build_classifier()

    def _build_classifier(self):
        """Build simple keyword-based domain classifier."""
        return {
            'legal': ['law', 'legal', 'attorney', 'court', 'litigation', 'counsel'],
            'medical': ['medical', 'health', 'hospital', 'clinic', 'doctor', 'patient'],
        }

    def detect_domain(self, text: str) -> str:
        """Detect domain from text."""
        text_lower = text.lower()

        for domain, keywords in self.domain_classifier.items():
            if any(kw in text_lower for kw in keywords):
                return domain

        return 'general'

    def encode(self, text: str) -> np.ndarray:
        """Encode text with domain-appropriate model."""
        domain = self.detect_domain(text)
        model = self.models.get(domain, self.models['general'])
        return model.encode([text], convert_to_numpy=True)[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch with domain-aware routing."""
        # Group by domain
        by_domain = {}
        for i, text in enumerate(texts):
            domain = self.detect_domain(text)
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append((i, text))

        # Encode each domain batch
        results = np.zeros((len(texts), 384))  # Assuming 384-dim

        for domain, items in by_domain.items():
            model = self.models.get(domain, self.models['general'])
            indices, domain_texts = zip(*items)
            embeddings = model.encode(list(domain_texts), convert_to_numpy=True)
            for idx, emb in zip(indices, embeddings):
                results[idx] = emb

        return results
```

---

## 6. Improved Concept Anchors

### 6.1 Problem

Current 31 anchors may not cover all relevant business domains.

### 6.2 Solution: Expanded & Data-Driven Anchors

```python
class DynamicConceptAnchors:
    """Data-driven concept anchor selection."""

    def __init__(self, company_names: List[str]):
        self.base_anchors = self._get_expanded_anchors()
        self.dynamic_anchors = self._extract_from_data(company_names)

    def _get_expanded_anchors(self) -> Dict[str, List[str]]:
        """Expanded anchor set."""
        return {
            "Geography_US": [
                "California", "Texas", "Florida", "New York", "Pennsylvania",
                "Illinois", "Ohio", "Georgia", "Michigan", "Arizona"
            ],
            "Geography_Intl": [
                "Canada", "Mexico", "UK", "Germany", "France", "Japan",
                "China", "Australia", "India", "Brazil"
            ],
            "Industry_Professional": [
                "Legal", "Accounting", "Consulting", "Engineering", "Architecture",
                "Medical", "Dental", "Veterinary"
            ],
            "Industry_Tech": [
                "Software", "Hardware", "Cloud", "AI", "Data", "Cybersecurity",
                "Telecommunications", "Internet"
            ],
            "Industry_Finance": [
                "Banking", "Insurance", "Investment", "Real Estate", "Mortgage",
                "Credit", "Wealth Management"
            ],
            "Industry_Healthcare": [
                "Hospital", "Clinic", "Pharmacy", "Laboratory", "Nursing",
                "Rehabilitation", "Mental Health"
            ],
            "Industry_Manufacturing": [
                "Automotive", "Aerospace", "Electronics", "Chemical", "Textile",
                "Food Processing", "Machinery"
            ],
            "Industry_Retail": [
                "Grocery", "Fashion", "Electronics", "Furniture", "Restaurant",
                "Hotel", "Entertainment"
            ],
            "Structure": [
                "Corporation", "LLC", "Partnership", "Non-Profit", "Government",
                "Sole Proprietor", "Cooperative"
            ],
            "Scale": [
                "Global", "National", "Regional", "Local", "Startup", "Enterprise"
            ],
            "Type": [
                "Manufacturer", "Distributor", "Retailer", "Service Provider",
                "Contractor", "Agency"
            ]
        }

    def _extract_from_data(self, company_names: List[str], top_n: int = 50) -> List[str]:
        """Extract frequent meaningful terms from company names as anchors."""
        from collections import Counter

        word_counts = Counter()

        for name in company_names:
            words = TextPreprocessor.clean_company_name(name).split()
            for word in words:
                if len(word) > 3 and word not in STOP_WORDS:
                    word_counts[word] += 1

        # Filter to meaningful terms (not too common, not too rare)
        total = len(company_names)
        meaningful = [
            word for word, count in word_counts.most_common(1000)
            if 0.001 < count / total < 0.05  # Between 0.1% and 5% frequency
        ]

        return meaningful[:top_n]

    def get_all_anchors(self) -> List[str]:
        """Get combined anchor list."""
        anchors = []
        for category_anchors in self.base_anchors.values():
            anchors.extend(category_anchors)
        anchors.extend(self.dynamic_anchors)
        return anchors
```

---

## 7. Score Calibration

### 7.1 Problem

Scores are not calibrated to actual match probability. A 90% score doesn't mean 90% confidence.

### 7.2 Solution: Probability Calibration

```python
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

class ScoreCalibrator:
    """Calibrate raw scores to probabilities."""

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit calibrator on labeled data.

        scores: Raw scores from matcher (0-1)
        labels: Binary labels (1 = correct match, 0 = incorrect)
        """
        self.calibrator.fit(scores, labels)
        self.is_fitted = True

    def calibrate(self, score: float) -> float:
        """Convert raw score to calibrated probability."""
        if not self.is_fitted:
            return score  # Return raw if not fitted

        return float(self.calibrator.predict([[score]])[0])

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate batch of scores."""
        if not self.is_fitted:
            return scores

        return self.calibrator.predict(scores)

    def get_calibration_curve(self, scores: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        """Get calibration curve data for visualization."""
        prob_true, prob_pred = calibration_curve(labels, scores, n_bins=n_bins)
        return prob_true, prob_pred

    def save(self, path: str):
        """Save calibrator to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)

    def load(self, path: str):
        """Load calibrator from file."""
        import pickle
        with open(path, 'rb') as f:
            self.calibrator = pickle.load(f)
        self.is_fitted = True
```

### 7.3 Integration

```python
class CompanyMatcher:
    def __init__(self, ...):
        ...
        self.score_calibrator = ScoreCalibrator()

        # Try to load pre-trained calibrator
        calibrator_path = os.path.join(self.cache_dir, 'score_calibrator.pkl')
        if os.path.exists(calibrator_path):
            self.score_calibrator.load(calibrator_path)

    def match_with_location(self, query, ...):
        results = self._match_impl(...)

        # Calibrate scores
        for result in results:
            raw_score = result['score']
            result['raw_score'] = raw_score
            result['score'] = self.score_calibrator.calibrate(raw_score)
            result['confidence'] = self._score_to_confidence(result['score'])

        return results

    def _score_to_confidence(self, calibrated_score: float) -> str:
        """Convert calibrated score to confidence label."""
        if calibrated_score >= 0.95:
            return 'very_high'
        elif calibrated_score >= 0.85:
            return 'high'
        elif calibrated_score >= 0.70:
            return 'medium'
        elif calibrated_score >= 0.50:
            return 'low'
        else:
            return 'very_low'
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)

1. **Feedback collection system** - Record user corrections
2. **Enhanced acronym engine** - Handle 2-letter, variants
3. **Score calibration** - Train on control set labels

### Phase 2: Learning (3-4 weeks)

4. **Adaptive weights** - Query-aware weight selection
5. **Feedback training pipeline** - Retrain from corrections
6. **A/B testing framework** - Compare algorithm variants

### Phase 3: Advanced (4-6 weeks)

7. **Fuzzy query processor** - Typo tolerance
8. **Domain ensemble** - Multiple embedding models
9. **Dynamic concept anchors** - Data-driven anchors
10. **Continuous learning** - Automated retraining

---

## 9. Evaluation Plan

### 9.1 Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Precision@1 | Top match correct | > 95% |
| Precision@5 | Correct in top 5 | > 99% |
| MRR | Mean reciprocal rank | > 0.90 |
| Calibration Error | |predicted - actual| | < 0.05 |
| User acceptance | Feedback accept rate | > 90% |

### 9.2 Evaluation Dataset

```python
# evaluation.py

class EvaluationDataset:
    """Curated evaluation dataset with ground truth."""

    def __init__(self):
        self.test_cases = [
            # Exact matches
            {'query': 'IBM', 'expected': 'IBM', 'type': 'exact'},

            # Acronym expansions
            {'query': 'ABA', 'expected': 'American Bar Association', 'type': 'acronym'},

            # Typos
            {'query': 'Microsft', 'expected': 'Microsoft', 'type': 'typo'},

            # Abbreviations
            {'query': 'Intl Business Machines', 'expected': 'International Business Machines', 'type': 'abbrev'},

            # Location variants
            {'query': 'Bank of America', 'city': 'Charlotte', 'expected_city': 'Charlotte', 'type': 'location'},

            # Semantic only
            {'query': 'Apple Inc', 'expected': 'Apple Inc.', 'type': 'semantic'},

            # Edge cases
            {'query': 'The Company', 'expected': None, 'type': 'ambiguous'},
        ]

    def evaluate(self, matcher) -> Dict:
        """Run evaluation."""
        results = {
            'precision_at_1': 0,
            'precision_at_5': 0,
            'mrr': 0,
            'by_type': {}
        }

        correct_1, correct_5, mrr_sum = 0, 0, 0.0

        for case in self.test_cases:
            matches = matcher.match_with_location(
                case['query'],
                city=case.get('city'),
                state=case.get('state'),
                top_k=10
            )

            expected = case['expected']
            if expected is None:
                continue

            # Check precision@1
            if matches and matches[0]['name'] == expected:
                correct_1 += 1

            # Check precision@5
            if any(m['name'] == expected for m in matches[:5]):
                correct_5 += 1

            # Calculate MRR
            for i, m in enumerate(matches):
                if m['name'] == expected:
                    mrr_sum += 1 / (i + 1)
                    break

        n = len([c for c in self.test_cases if c['expected'] is not None])
        results['precision_at_1'] = correct_1 / n
        results['precision_at_5'] = correct_5 / n
        results['mrr'] = mrr_sum / n

        return results
```

---

## 10. Success Criteria

1. **Control set accuracy > 97%** (current baseline needed)
2. **User feedback acceptance rate > 90%**
3. **Calibration error < 5%**
4. **Zero regressions** on existing high-confidence matches
5. **Handles 90%+ of acronyms correctly**
6. **Typo tolerance** for 1-2 character errors

---

*End of Proposal 2*
