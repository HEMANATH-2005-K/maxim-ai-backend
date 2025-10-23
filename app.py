from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from textblob import TextBlob
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Flutter connection

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Storage for conversations
conversation_history = {}

class EmotionalAI:
    @staticmethod
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.3:
            return {"sentiment": "positive", "emoji": "😊", "score": polarity}
        elif polarity < -0.3:
            return {"sentiment": "negative", "emoji": "😔", "score": polarity}
        else:
            return {"sentiment": "neutral", "emoji": "😐", "score": polarity}
    
    @staticmethod
    def get_emotional_response(sentiment_data, user_message):
        base_prompts = {
            "positive": "The user seems happy! Be enthusiastic and celebratory. Use emojis like 😊🎉🌟",
            "negative": "The user seems upset. Be extra empathetic, supportive, and caring. Use comforting emojis like 💙🤗🌧️→🌈",
            "neutral": "The user is neutral. Be engaging, curious, and friendly. Use emojis like 🤔💭✨"
        }
        return base_prompts.get(sentiment_data["sentiment"], "Be friendly and helpful.")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "AI Server is running! 🚀", "timestamp": time.time()})

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"📨 Received: {user_message}")
        
        # Analyze sentiment
        sentiment = EmotionalAI.analyze_sentiment(user_message)
        emotional_prompt = EmotionalAI.get_emotional_response(sentiment, user_message)
        
        # Maintain conversation history
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        conversation_history[user_id].append({"role": "user", "content": user_message})
        
        # Keep only last 6 messages
        if len(conversation_history[user_id]) > 6:
            conversation_history[user_id] = conversation_history[user_id][-6:]
        
        # Prepare messages for GPT
        system_prompt = f"""You are EmpathAI - a deeply empathetic, supportive AI friend. 

{emotional_prompt}

Key personality:
- Warm and genuinely caring
- Use appropriate emojis
- Remember conversation context
- Be conversational but meaningful
- Keep responses under 3 sentences
- Sometimes ask thoughtful questions

User's emotion: {sentiment['sentiment']} {sentiment['emoji']}
"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[user_id])
        
        # Call OpenAI API
        start_time = time.time()
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.8,
        )
        
        ai_response = response.choices[0].message.content
        response_time = time.time() - start_time
        
        # Add AI response to history
        conversation_history[user_id].append({"role": "assistant", "content": ai_response})
        
        print(f"🤖 Response time: {response_time:.2f}s")
        
        return jsonify({
            "reply": ai_response,
            "sentiment": sentiment,
            "response_time": response_time,
            "message_id": len(conversation_history[user_id]),
            "timestamp": time.time()
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "reply": "I'm having trouble connecting right now, but I'm still here for you! 💙 What's on your mind?",
            "sentiment": {"sentiment": "neutral", "emoji": "🤖", "score": 0},
            "error": str(e)
        }), 500

@app.route('/conversation/<user_id>', methods=['GET'])
def get_conversation(user_id):
    return jsonify({
        "user_id": user_id,
        "messages": conversation_history.get(user_id, []),
        "total_messages": len(conversation_history.get(user_id, []))
    })

if __name__ == '__main__':
    print("🚀 Starting Emotional AI Server...")
    print("💬 Chat Endpoint: http://localhost:5000/chat")
    print("❤️ Health Check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)