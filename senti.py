from flask import Flask, request, jsonify
from textblob import TextBlob
import re
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from http import HTTPStatus

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='sentiment_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up rate limiting with explicit in-memory storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Explicitly use in-memory storage
)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()


# Function to clean the text
def clean_text(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text.strip()


# Sentiment analysis endpoint
@app.route('/analyze-sentiment', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_sentiment():
    try:
        if not request.is_json:
            logger.warning("Invalid content type: %s", request.content_type)
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), HTTPStatus.BAD_REQUEST

        data = request.get_json()
        if not data or 'text' not in data:
            logger.warning("Missing 'text' field in request body")
            return jsonify({
                'error': 'No text provided. Please include "text" in the request body.'
            }), HTTPStatus.BAD_REQUEST

        text = data['text']
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided")
            return jsonify({
                'error': 'Text cannot be empty.'
            }), HTTPStatus.BAD_REQUEST

        if len(text) > 1000:
            logger.warning("Text too long: %d characters", len(text))
            return jsonify({
                'error': 'Text is too long. Maximum length is 1000 characters.'
            }), HTTPStatus.BAD_REQUEST

        cleaned_text = clean_text(text)
        if not cleaned_text:
            logger.warning("Text is empty after cleaning: %s", text)
            return jsonify({
                'error': 'Text is empty after cleaning.'
            }), HTTPStatus.BAD_REQUEST

        textblob_analysis = TextBlob(cleaned_text)
        textblob_polarity = textblob_analysis.sentiment.polarity
        textblob_subjectivity = textblob_analysis.sentiment.subjectivity

        vader_scores = vader_analyzer.polarity_scores(cleaned_text)
        vader_compound = vader_scores['compound']

        combined_polarity = (textblob_polarity + vader_compound) / 2
        if combined_polarity > 0.1:
            sentiment = 'positive'
        elif combined_polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_compound': vader_compound,
            'combined_polarity': combined_polarity
        }

        logger.info("Sentiment analysis successful: %s", result)
        return jsonify(result), HTTPStatus.OK

    except ValueError as ve:
        logger.error("ValueError during sentiment analysis: %s", str(ve))
        return jsonify({
            'error': str(ve)
        }), HTTPStatus.BAD_REQUEST

    except Exception as e:
        logger.error("Unexpected error during sentiment analysis: %s", str(e))
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running'
    }), HTTPStatus.OK

# ... (previous code remains the same)

# Endpoint to get rate limit status
@app.route('/rate-limit-status', methods=['GET'])
def rate_limit_status():
    return jsonify({
        'message': 'Rate limit status',
        'default_limits': limiter.limiter.default_limits(),
        'current_limits': limiter.limiter.get_window_stats(limiter.current_limit, get_remote_address())
    }), HTTPStatus.OK

# Endpoint to test logging
@app.route('/test-log', methods=['GET'])
def test_log():
    logger.info("This is a test log message")
    return jsonify({
        'message': 'Logged a test message. Check sentiment_api.log for details.'
    }), HTTPStatus.OK

# Endpoint to analyze multiple texts
@app.route('/analyze-sentiment-batch', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_sentiment_batch():
    try:
        if not request.is_json:
            logger.warning("Invalid content type: %s", request.content_type)
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), HTTPStatus.BAD_REQUEST

        data = request.get_json()
        if not data or 'texts' not in data:
            logger.warning("Missing 'texts' field in request body")
            return jsonify({
                'error': 'No texts provided. Please include "texts" (list) in the request body.'
            }), HTTPStatus.BAD_REQUEST

        texts = data['texts']
        if not isinstance(texts, list):
            logger.warning("Invalid 'texts' field: not a list")
            return jsonify({
                'error': '"texts" must be a list of strings.'
            }), HTTPStatus.BAD_REQUEST

        if len(texts) > 10:  # Limit batch size
            logger.warning("Too many texts in batch: %d", len(texts))
            return jsonify({
                'error': 'Too many texts. Maximum batch size is 10.'
            }), HTTPStatus.BAD_REQUEST

        results = []
        for text in texts:
            if not isinstance(text, str):
                results.append({'text': str(text), 'error': 'Text must be a string'})
                continue

            if not text or len(text.strip()) == 0:
                results.append({'text': text, 'error': 'Text cannot be empty'})
                continue

            if len(text) > 1000:
                results.append({'text': text, 'error': 'Text is too long. Maximum length is 1000 characters.'})
                continue

            cleaned_text = clean_text(text)
            if not cleaned_text:
                results.append({'text': text, 'error': 'Text is empty after cleaning'})
                continue

            textblob_analysis = TextBlob(cleaned_text)
            textblob_polarity = textblob_analysis.sentiment.polarity
            textblob_subjectivity = textblob_analysis.sentiment.subjectivity

            vader_scores = vader_analyzer.polarity_scores(cleaned_text)
            vader_compound = vader_scores['compound']

            combined_polarity = (textblob_polarity + vader_compound) / 2
            if combined_polarity > 0.1:
                sentiment = 'positive'
            elif combined_polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            results.append({
                'original_text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_compound': vader_compound,
                'combined_polarity': combined_polarity
            })

        logger.info("Batch sentiment analysis successful: %d texts processed", len(results))
        return jsonify({'results': results}), HTTPStatus.OK

    except Exception as e:
        logger.error("Unexpected error during batch sentiment analysis: %s", str(e))
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)