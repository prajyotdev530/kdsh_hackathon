from flask import Flask, render_template, request, jsonify
import os
import sys
from consistency_checker import ConsistencyChecker
import traceback

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global variables for the consistency checker
checker = None

def initialize_checker():
    """Initialize the consistency checker"""
    global checker
    try:
        checker = ConsistencyChecker()

        # Load embeddings
        embeddings_dir = os.path.join(os.path.dirname(__file__), 'embeddings_cache')
        os.makedirs(embeddings_dir, exist_ok=True)

        # Load book embeddings
        checker.load_embeddings('In Search of the Castaways', './In_Search_of_the_Castaways_embeddings.pkl')
        checker.load_embeddings('The Count of Monte Cristo', './The_Count_of_Monte_Cristo_embeddings.pkl')

        print("✅ Consistency checker initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing checker: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Home page with the consistency checker interface"""
    return render_template('index.html')

@app.route('/check_consistency', methods=['POST'])
def check_consistency():
    """API endpoint to check statement consistency"""
    try:
        if not checker:
            return jsonify({
                'error': 'Consistency checker not initialized',
                'consistent': False,
                'confidence': 0.0
            }), 500

        # Get form data
        book_name = request.form.get('book_name', '')
        character = request.form.get('character', '')
        statement = request.form.get('statement', '')

        if not all([book_name, character, statement]):
            return jsonify({
                'error': 'Missing required fields',
                'consistent': False,
                'confidence': 0.0
            }), 400

        # Get relevant chunks for additional context
        relevant_chunks = checker.get_relevant_chunks(book_name, statement, char_name=character, top_k=3)

        # Check consistency
        prediction = checker.check_consistency(book_name, character, statement)

        # Calculate confidence (simplified - you could enhance this)
        avg_similarity = sum(float(chunk['similarity']) for chunk in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0.0
        confidence = min(0.95, avg_similarity * 1.2)  # Scale similarity to confidence

        # Prepare response
        response = {
            'consistent': prediction == 'consistent',
            'prediction': prediction,
            'confidence': round(confidence * 100, 1),
            'book': book_name,
            'character': character,
            'statement': statement,
            'relevant_chunks': [
                {
                    'text': chunk['chunk'][:300] + '...' if len(chunk['chunk']) > 300 else chunk['chunk'],
                    'similarity': round(float(chunk['similarity']), 3)
                } for chunk in relevant_chunks[:3]
            ]
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in check_consistency: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'consistent': False,
            'confidence': 0.0
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'checker_initialized': checker is not None,
        'models_loaded': len(checker.stores) if checker else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Narrative Consistency Checker Web App")
    print("=" * 60)

    # Initialize the checker
    if initialize_checker():
        print("🌐 Starting Flask server on http://localhost:8000")
        print("📖 Open your browser and navigate to the URL above")
        print("❌ Press Ctrl+C to stop the server")
        print("=" * 60)

        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("❌ Failed to initialize consistency checker. Exiting.")
