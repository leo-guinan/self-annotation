# startup_mdp_trainer.py

import streamlit as st

import json

import time

import random

import numpy as np

from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass

import pandas as pd

from datetime import datetime

import ollama

from supabase import create_client, Client



# ============================================================================

# CONFIGURATION

# ============================================================================



@dataclass

class LevelConfig:

    name: str

    goal: int

    skill: str

    reward_xp: int



LEVELS = [

    LevelConfig("Rookie", 5, "Identify Result archetypes", 50),

    LevelConfig("Explorer", 10, "Distinguish Pivot vs Insight", 100),

    LevelConfig("Analyst", 15, "Extract numerical state variables", 150),

    LevelConfig("Architect", 20, "Build complete decision chains", 200),

    LevelConfig("Sensei", 30, "Annotate with counterfactuals", 300),

]



SUBGAMES = {

    "product": "Changing the product to reduce friction",

    "marketing": "Educating market about differentiation", 

    "sales": "Converting leads to revenue",

    "finance": "Managing runway and capital",

    "engineering": "Reducing system uncertainty",

    "prophecy": "Predicting future trends"

}



# Ollama integration for LLM calls

# Supabase integration for Community Archive
SUPABASE_URL = "https://fabxmporizzqflnftavs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)



# ============================================================================

# SESSION STATE INITIALIZATION

# ============================================================================



def init_session_state():

    """Initialize all session state variables"""

    if "user_id" not in st.session_state:

        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"

    

    if "annotations" not in st.session_state:

        st.session_state.annotations = []

    

    if "current_level" not in st.session_state:

        st.session_state.current_level = 0

        st.session_state.level_progress = 0

    

    if "xp" not in st.session_state:

        st.session_state.xp = 0

    

    if "model_trained" not in st.session_state:

        st.session_state.model_trained = False

    

    if "writing_drafts" not in st.session_state:

        st.session_state.writing_drafts = []

    

    if "tutorial_completed" not in st.session_state:

        st.session_state.tutorial_completed = False
    
    if "user_tweets" not in st.session_state:
        st.session_state.user_tweets = []
    
    if "current_username" not in st.session_state:
        st.session_state.current_username = None



# ============================================================================

# ONBOARDING & ANALYSIS

# ============================================================================



def onboarding_screen():

    st.title("🚀 Train Your Decision Twin")

    st.markdown("""

    ### Welcome to the Startup MDP Annotation Gym

    

    **Your Mission**: Transform your tweets and thoughts into a personalized AI assistant 

    that understands how you make decisions.

    

    **Why?**: 

    - Learn systems thinking via MDPs

    - Build a model that writes like you

    - Join a community of founder-annotators

    - All hosted locally, zero cost

    """)

    

    # Simulated tweet analysis

    handle = st.text_input("🔗 Enter your X handle (optional)", 

                          help="We'll analyze your last 100 tweets to find decision patterns")

    

    if handle:

        analyze_user_tweets(handle)

    

    # Start tutorial

    if st.button("🎓 Start Tutorial", type="primary"):

        st.session_state.tutorial_completed = False

        st.session_state.current_page = "tutorial"



def fetch_user_tweets_from_supabase(username: str) -> Tuple[List[Dict], Optional[str]]:
    """Fetch all tweets for a user from Supabase
    
    Returns:
        Tuple of (tweets list, error message if any)
    """
    try:
        supabase = get_supabase_client()
        
        # First, get the account_id from username
        account_response = supabase.table('account').select('account_id').eq('username', username.lower()).execute()
        
        if not account_response.data or len(account_response.data) == 0:
            return [], f"No account found for username: {username}"
        
        account_id = account_response.data[0]['account_id']
        
        # Fetch all tweets for this account (with pagination)
        all_tweets = []
        page_size = 1000
        offset = 0
        
        while True:
            tweets_response = supabase.table('tweets').select('*').eq('account_id', account_id).range(offset, offset + page_size - 1).execute()
            
            if not tweets_response.data or len(tweets_response.data) == 0:
                break
            
            all_tweets.extend(tweets_response.data)
            
            if len(tweets_response.data) < page_size:
                break
            
            offset += page_size
        
        return all_tweets, None
    
    except Exception as e:
        return [], f"Error fetching tweets: {str(e)}"


def analyze_user_tweets(handle: str):

    """Analyze user's tweets from Community Archive"""

    with st.spinner(f"Analyzing @{handle}'s tweets from Community Archive..."):
        
        # Fetch real tweets from Supabase
        tweets, error = fetch_user_tweets_from_supabase(handle)
        
        if error:
            st.error(f"⚠️ {error}")
            return
        
        if not tweets:
            st.warning(f"⚠️ No tweets found for @{handle}. Make sure the username is correct and exists in the Community Archive.")
            return
        
        # Store tweets in session state
        st.session_state.user_tweets = tweets
        st.session_state.current_username = handle.lower()
        
        # Debug: Check what fields are in the first tweet
        if tweets:
            with st.expander("🔍 Debug: Tweet Structure", expanded=False):
                st.json(list(tweets[0].keys())[:10])  # Show first 10 keys
                # Try to find the text field
                text_fields = ['text', 'full_text', 'content', 'tweet_text', 'body']
                found_text_field = None
                for field in text_fields:
                    if field in tweets[0]:
                        found_text_field = field
                        st.info(f"Found text field: '{field}'")
                        st.text(f"Sample: {str(tweets[0].get(field, ''))[:200]}")
                        break
                if not found_text_field:
                    st.warning("Could not find text field. Available fields shown above.")
        
        # Analyze tweets for decision-rich content
        # Expanded decision indicators
        decision_indicators = [
            # Numbers and metrics
            '→', '->', '%', 'percent', 'x', 'times', '2x', '3x', '4x', '5x',
            # Action words
            'improved', 'increased', 'decreased', 'changed', 'pivoted', 'decided',
            'result', 'outcome', 'shipped', 'launched', 'built', 'created',
            'reduced', 'grew', 'scaled', 'optimized', 'fixed', 'solved',
            # Comparison words
            'from', 'to', 'before', 'after', 'was', 'now', 'became',
            # Decision words
            'decision', 'choice', 'strategy', 'tactic', 'approach', 'method',
            # Metrics
            'users', 'revenue', 'growth', 'conversion', 'retention', 'churn',
            'mrr', 'arr', 'cac', 'ltv', 'activation', 'engagement'
        ]
        
        decision_tweets = []
        
        # Try different possible text field names
        text_field = None
        for field in ['text', 'full_text', 'content', 'tweet_text', 'body']:
            if tweets and field in tweets[0]:
                text_field = field
                break
        
        if not text_field:
            st.error("Could not find tweet text field in database. Please check the tweet structure.")
            return
        
        # Store text field name in session state for later use
        st.session_state.tweet_text_field = text_field
        
        for tweet in tweets:
            tweet_text = str(tweet.get(text_field, '')).lower()
            if tweet_text and any(indicator.lower() in tweet_text for indicator in decision_indicators):
                decision_tweets.append(tweet)
        
        decision_count = len(decision_tweets)
        total_count = len(tweets)
        activation_rate = min(decision_count / 50, 1.0) if total_count > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tweets", total_count)
        
        with col2:
            st.metric("Decision-Rich Tweets", decision_count)
        
        with col3:
            st.metric("Training Readiness", f"{activation_rate:.0%}")
        
        if decision_count >= 20:
            st.success("🎯 **Perfect!** You have enough tweets to train a high-quality model.")
            st.info("**Next**: Complete 20 annotations → Train your Decision Twin")
        else:
            st.warning(f"💡 Need ~{max(0, 20 - decision_count)} more decision-rich tweets for best results.")
        
        # Show sample tweets
        if decision_tweets:
            with st.expander("📝 Sample Decision Tweets", expanded=False):
                for i, tweet in enumerate(decision_tweets[:5]):
                    tweet_text = str(tweet.get(text_field or 'text', ''))[:100]
                    st.text(f"{i+1}. {tweet_text}...")
        else:
            # Show sample of all tweets to help debug
            with st.expander("🔍 Debug: Sample of All Tweets", expanded=False):
                for i, tweet in enumerate(tweets[:3]):
                    st.json({k: str(v)[:100] for k, v in list(tweet.items())[:5]})



def tutorial_screen():

    st.title("📚 The State-Action-Outcome Framework")

    

    # Interactive breakdown

    st.markdown("### Let's break down a real decision")

    

    example_tweets = [

        "Simplified onboarding, activation doubled from 15% to 30%",

        "Markets are shifting, we're pivoting from B2C to B2B",

        "Sales tip: following up in 24h increased close rate by 40%"

    ]

    

    selected = st.selectbox("Choose an example tweet", example_tweets)

    

    # Three-column visualization

    col1, col2, col3 = st.columns(3)

    

    with col1:

        st.subheader("🎯 State (Before)")

        st.markdown("What was true *before* the action?")

        if "doubled" in selected:

            friction = st.slider("User Friction", 0.0, 1.0, 0.85)

            st.caption(f"State energy cost: {friction:.1f} units")

            st.info("High friction = users need lots of energy to succeed")

    

    with col2:

        st.subheader("⚡ Action (What You Did)")

        st.markdown("What specific action was taken?")

        if "doubled" in selected:

            action = st.text_area("Action", "remove 3 features from onboarding")

            st.metric("Leverage", 2.0, help="Future time saved per user")

    

    with col3:

        st.subheader("📈 Outcome (After)")

        st.markdown("What measurable changed?")

        if "doubled" in selected:

            outcome = st.text_input("Outcome", "activation: 15% → 30%")

            timeline = st.text_input("Time to result", "14 days")

    

    # Calculate reward

    if st.button("💎 Calculate Decision Quality"):

        reward = random.uniform(0.15, 0.35)  # Mock calculation

        st.success(f"**Decision Quality Score**: {reward:.2f}")

        st.info("Higher score = clearer state-action-outcome + bigger impact")

        

        if reward > 0.25:

            st.balloons()

            st.session_state.xp += 10

    

    # Complete tutorial

    if st.button("✅ Complete Tutorial", type="primary"):

        st.session_state.tutorial_completed = True

        st.session_state.xp += 25

        st.success("Tutorial complete! Unlocked Annotation Gym.")



# ============================================================================

# ANNOTATION GYM

# ============================================================================



def annotation_gym():

    st.title("🏋️ Annotation Gym")

    

    # Level HUD

    render_level_hud()

    

    # Daily challenge

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Daily Challenge")

        challenge = get_daily_challenge()

        st.write(f"**Task**: {challenge['task']}")

        st.caption(f"Hint: {challenge['hint']}")

        st.metric("Reward", f"+{challenge['reward_xp']} XP")

    

    with col2:

        st.subheader("📊 Your Progress")

        st.metric("Total Annotations", len(st.session_state.annotations))

        st.metric("Total XP", st.session_state.xp)

    

    # Annotation interface

    st.markdown("---")

    st.subheader("📤 Annotate Your Next Decision")

    

    # Get tweet to annotate (from user's tweets or sample)
    if st.session_state.user_tweets:
        # Use stored text field name or detect it
        text_field = st.session_state.get('tweet_text_field')
        if not text_field:
            for field in ['text', 'full_text', 'content', 'tweet_text', 'body']:
                if st.session_state.user_tweets and field in st.session_state.user_tweets[0]:
                    text_field = field
                    st.session_state.tweet_text_field = field
                    break
        if not text_field:
            text_field = 'text'  # Fallback
        
        # Filter for decision-rich tweets (expanded indicators)
        decision_indicators = [
            '→', '->', '%', 'percent', 'x', 'times', '2x', '3x', '4x', '5x',
            'improved', 'increased', 'decreased', 'changed', 'pivoted', 'decided',
            'result', 'outcome', 'shipped', 'launched', 'built', 'created',
            'reduced', 'grew', 'scaled', 'optimized', 'fixed', 'solved',
            'from', 'to', 'before', 'after', 'was', 'now', 'became',
            'decision', 'choice', 'strategy', 'tactic', 'approach', 'method',
            'users', 'revenue', 'growth', 'conversion', 'retention', 'churn',
            'mrr', 'arr', 'cac', 'ltv', 'activation', 'engagement'
        ]
        
        decision_tweets = [
            t for t in st.session_state.user_tweets 
            if any(indicator.lower() in str(t.get(text_field, '')).lower() for indicator in decision_indicators)
        ]
        
        # Get tweets that haven't been annotated yet
        annotated_ids = {a.get('tweet_id') for a in st.session_state.annotations if 'tweet_id' in a}
        unannotated_tweets = [t for t in decision_tweets if t.get('id') not in annotated_ids]
        
        if unannotated_tweets:
            # Select a random unannotated tweet
            if "current_tweet_index" not in st.session_state:
                st.session_state.current_tweet_index = 0
            
            # Tweet selector
            tweet_options = {i: f"Tweet {i+1}: {t.get('text', '')[:60]}..." for i, t in enumerate(unannotated_tweets[:50])}
            selected_idx = st.selectbox(
                f"Select tweet to annotate ({len(unannotated_tweets)} available)",
                range(len(unannotated_tweets[:50])),
                format_func=lambda x: tweet_options[x],
                key="tweet_selector"
            )
            
            current_tweet = unannotated_tweets[selected_idx]
            # Get text from the correct field
            text_field = None
            for field in ['text', 'full_text', 'content', 'tweet_text', 'body']:
                if field in current_tweet:
                    text_field = field
                    break
            default_text = current_tweet.get(text_field or 'text', '')
            tweet_id = current_tweet.get('id')
            tweet_timestamp = current_tweet.get('created_at', '')
        else:
            default_text = "All decision-rich tweets have been annotated! Add more tweets or use a sample."
            tweet_id = None
            tweet_timestamp = None
    else:
        # Fallback to sample tweet
        default_text = "Spent 2 weeks refactoring our auth system. Tech debt was blocking all feature work. Velocity up 3x now."
        tweet_id = None
        tweet_timestamp = None
        st.info("💡 Enter your X handle in the Onboarding tab to load your real tweets!")
    
    tweet_text = st.text_area("Tweet to annotate", default_text, height=120)

    

    # Subgame selection

    subgame = st.radio("Which subgame?", list(SUBGAMES.keys()), 

                      format_func=lambda x: f"{x.title()}: {SUBGAMES[x]}")

    

    # Dynamic form based on subgame

    if subgame == "engineering":

        col1, col2 = st.columns(2)

        with col1:

            uncertainty = st.slider("Requirement Uncertainty", 0.0, 1.0, 0.7)

            entropy = st.slider("System Entropy", 0.0, 1.0, 0.8)

        with col2:

            debt_cost = st.number_input("Tech Debt Cost (hrs/week)", 0, 100, 20)

            velocity_impact = st.slider("Velocity Impact", 0.0, 5.0, 3.0)

    

    elif subgame == "product":

        col1, col2 = st.columns(2)

        with col1:

            friction = st.slider("User Friction", 0.0, 1.0, 0.7)

            leverage = st.slider("Current Leverage", 0.0, 5.0, 1.5)

        with col2:

            activation = st.slider("Activation Rate", 0.0, 1.0, 0.3)

            ttv_hours = st.number_input("Time to Value (hours)", 1, 168, 48)

    

    # Action details

    action_taken = st.text_input("Action Taken", "refactor_auth_system")

    

    # Evidence selection

    st.markdown("#### 📚 Evidence (select text spans)")

    words = tweet_text.split()

    evidence_options = st.multiselect("Key phrases", words[:20], 

                                     default=words[:3] if words else None)

    

    # Submit

    col1, col2 = st.columns([1, 2])

    with col1:

        if st.button("✅ Submit Annotation", type="primary"):

            # Save annotation

            annotation = {

                "id": len(st.session_state.annotations),

                "subgame": subgame,

                "text": tweet_text,

                "state_before": {"friction": friction, "leverage": leverage},

                "action_taken": action_taken,

                "evidence_quotes": evidence_options,

                "timestamp": datetime.now().isoformat(),

                "confidence": random.uniform(0.7, 0.95)

            }
            
            # Add tweet metadata if available

            if tweet_id:

                annotation["tweet_id"] = tweet_id

                annotation["tweet_timestamp"] = tweet_timestamp

                annotation["username"] = st.session_state.current_username

            st.session_state.annotations.append(annotation)

            

            # Update progress

            st.session_state.level_progress += 1

            st.session_state.xp += 10

            

            # Check level up

            check_level_up()

            

            st.success(f"Annotation saved! +10 XP")

            st.balloons()

            

            # Show what AI would have extracted

            with st.expander("🤖 Compare with AI"):

                extraction_prompt = f"""Extract structured decision traces from this tweet in JSON format:

{tweet_text}

Return a JSON array with this structure:
{{
  "subgame": "product|marketing|sales|finance|engineering|prophecy",
  "timestep": {{"week": number, "funding_stage": "string"}},
  "state_before": {{"friction": 0.0-1.0, "leverage": 0.0-5.0}},
  "action_taken": "string describing the action",
  "observed_outcome": {{"metric": value}},
  "estimated_reward": 0.0-1.0,
  "confidence": 0.0-1.0,
  "evidence_quotes": ["key phrases from tweet"]
}}"""

                try:

                    response = ollama.chat(

                        model="llama3.2:13b",

                        messages=[{"role": "user", "content": extraction_prompt}],

                        format="json",

                        options={"temperature": 0.1}

                    )

                    ai_content = response["message"]["content"]

                    # Try to parse and display as JSON

                    try:

                        ai_json = json.loads(ai_content)

                        st.json(ai_json)

                    except json.JSONDecodeError:

                        # If not valid JSON, display as text

                        st.text(ai_content)

                except Exception as e:

                    st.error(f"Error calling Ollama: {str(e)}")

                    st.info("Make sure Ollama is running and llama3.2:13b model is installed.")

    

    with col2:

        if st.button("❌ Skip (Not a Decision)"):

            st.info("Skipped!")

            st.session_state.xp += 1  # Small reward for not annotating junk



def render_level_hud():

    """Show current level and progress"""

    current = LEVELS[st.session_state.current_level]

    

    col1, col2, col3 = st.columns(3)

    

    with col1:

        st.metric("Level", current.name)

    

    with col2:

        progress = st.session_state.level_progress

        target = current.goal

        st.metric("Progress", f"{progress}/{target}")

    

    with col3:

        st.metric("Total XP", st.session_state.xp)

    

    # Progress bar

    st.progress(min(progress / target, 1.0))

    

    # Skill focus

    st.info(f"🎯 **Current Focus**: {current.skill}")

    

    # Next level preview

    if progress >= target:

        if st.session_state.current_level < len(LEVELS) - 1:

            next_level = LEVELS[st.session_state.current_level + 1]

            st.success(f"🎉 Level complete! Next: {next_level.name}")

            if st.button("Unlock Next Level"):

                st.session_state.current_level += 1

                st.session_state.level_progress = 0

                st.session_state.xp += current.reward_xp

                st.balloons()

        else:

            st.success("🏆 You've mastered annotation!")



def check_level_up():

    """Check if user should level up"""

    current = LEVELS[st.session_state.current_level]

    if st.session_state.level_progress >= current.goal:

        st.success(f"🎉 Level {current.name} complete! +{current.reward_xp} XP")



def get_daily_challenge() -> Dict:

    """Generate daily challenge based on current level"""

    challenges = {

        "Rookie": {

            "task": "Find 3 'Result' archetype tweets (look for '→', '%', '2x')",

            "hint": "Results show measurable change: '15% → 30%'",

            "reward_xp": 50

        },

        "Explorer": {

            "task": "Distinguish between Pivot vs Insight decisions",

            "hint": "Pivot = changing direction, Insight = deeper understanding",

            "reward_xp": 75

        },

        "Analyst": {

            "task": "Extract numerical state variables from 5 tweets",

            "hint": "Convert 'slow' → friction=0.8, 'fast' → friction=0.2",

            "reward_xp": 100

        }

    }

    

    current = LEVELS[st.session_state.current_level].name

    return challenges.get(current, {"task": "Keep annotating!", "hint": "Practice makes perfect", "reward_xp": 25})



# ============================================================================

# MODEL TRAINING

# ============================================================================



def model_training_screen():

    st.title("🧬 Train Your Decision Twin")

    

    # Prerequisites check

    if len(st.session_state.annotations) < 5:

        st.warning("⚠️ Need at least 5 annotations to train a model.")

        st.info(f"Current: {len(st.session_state.annotations)}/5")

        return

    

    # Dataset quality dashboard

    col1, col2, col3 = st.columns(3)

    

    with col1:

        st.metric("Training Examples", len(st.session_state.annotations))

    

    with col2:

        avg_confidence = np.mean([a.get("confidence", 0.7) for a in st.session_state.annotations])

        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    

    with col3:

        subgame_dist = pd.DataFrame(st.session_state.annotations)["subgame"].value_counts()

        st.metric("Subgames Covered", len(subgame_dist))

    

    # Subgame distribution

    st.subheader("Your Decision Distribution")

    st.bar_chart(subgame_dist)

    

    # Training config

    st.subheader("Training Configuration")

    

    col1, col2 = st.columns(2)

    

    with col1:

        epochs = st.slider("Training Epochs", 1, 10, 3,

                          help="More epochs = more memorized, less generalizable")

    

    with col2:

        creativity = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3,

                              help="Higher = more diverse, unexpected suggestions")

    

    # Train button

    if st.button("🚀 Train Model", type="primary", use_container_width=True):

        with st.spinner("Training local model (this takes 5-10 minutes)...") as status:

            train_stats = simulate_model_training(

                annotations=st.session_state.annotations,

                epochs=epochs,

                temperature=creativity

            )

        

        st.success("✅ Training complete!")

        st.balloons()

        

        st.session_state.model_trained = True

        st.session_state.train_stats = train_stats

        

        # Show training metrics

        st.subheader("Training Results")

        st.line_chart(train_stats["loss_curve"])

        

        # Model card

        st.info(f"""

        **Your Decision Twin Stats**:

        - Training examples: {train_stats["n_examples"]}

        - Final loss: {train_stats["final_loss"]:.3f}

        - Est. inference time: {train_stats["inference_ms"]}ms

        

        **To use**: Your model is now active in the Writing Assistant tab.

        """)

        

        st.session_state.xp += 150

    

    # Export data

    st.subheader("Export Your Data")

    col1, col2 = st.columns(2)

    

    with col1:

        json_export = json.dumps(st.session_state.annotations, indent=2)

        st.download_button("📥 Download Annotations (JSON)", 

                          json_export, "my_annotations.json")

    

    with col2:

        csv_export = pd.DataFrame(st.session_state.annotations).to_csv(index=False)

        st.download_button("📊 Download Annotations (CSV)", 

                          csv_export, "my_annotations.csv")



def simulate_model_training(annotations: List[Dict], epochs: int, temperature: float) -> Dict:

    """Simulate model training (replace with actual fine-tuning)"""

    

    # Mock training curve

    loss_curve = [0.8 - (i * 0.05) + random.uniform(-0.02, 0.02) for i in range(epochs * 10)]

    

    time.sleep(2)  # Simulate training time

    

    return {

        "n_examples": len(annotations),

        "final_loss": loss_curve[-1],

        "inference_ms": 200,

        "loss_curve": loss_curve,

        "temperature": temperature

    }



# ============================================================================

# WRITING ASSISTANT

# ============================================================================



def writing_assistant_screen():

    st.title("✍️ Writing Assistant with Decision Twin")

    

    if not st.session_state.model_trained:

        st.warning("⚠️ Train your Decision Twin first!")

        if st.button("Go to Training"):

            st.session_state.current_page = "training"

        return

    

    st.markdown("### Generate tweets based on your past decision patterns")

    

    # Intent gathering

    col1, col2 = st.columns(2)

    

    with col1:

        intent = st.text_area("What do you want to write about?",

                             placeholder="e.g., A lesson about simplifying a product",

                             height=100)

    

    with col2:

        context = st.text_area("Context (private details)",

                              placeholder="Numbers, timeline, specific examples...",

                              height=100)

    

    # Generation controls

    col1, col2, col3 = st.columns(3)

    

    with col1:

        examples = st.slider("Examples to generate", 1, 5, 3)

    

    with col2:

        max_length = st.slider("Max length (words)", 20, 100, 45)

    

    with col3:

        archetype_filter = st.multiselect(

            "Filter by archetype",

            ["result", "pivot", "insight", "mistake", "prediction"],

            default=["result", "insight"]

        )

    

    # Generate

    if st.button("🤖 Generate Suggestions", type="primary", use_container_width=True):

        with st.spinner("Your Decision Twin is thinking..."):

            suggestions = generate_tweet_suggestions(

                user_id=st.session_state.user_id,

                intent=intent,

                context=context,

                n=examples,

                max_length=max_length,

                archetypes=archetype_filter,

                stats=st.session_state.train_stats

            )

        

        st.session_state.writing_drafts = suggestions

        

        st.success(f"✅ Generated {len(suggestions)} suggestions")

    

    # Display suggestions

    if st.session_state.writing_drafts:

        st.markdown("---")

        st.subheader("📝 Your Suggestions")

        

        for i, draft in enumerate(st.session_state.writing_drafts):

            with st.expander(f"Draft {i+1}: {draft['archetype']} archetype (conf: {draft['confidence']:.1%})", expanded=True):

                # Editable tweet

                edited = st.text_area(

                    "Edit before posting",

                    draft["tweet"],

                    height=80,

                    key=f"draft_{i}"

                )

                

                # Metadata

                col1, col2, col3, col4 = st.columns(4)

                

                with col1:

                    if st.button("📤 Post", key=f"post_{i}"):

                        st.success("Posted! (simulated)")

                

                with col2:

                    if st.button("💾 Save", key=f"save_{i}"):

                        st.info("Saved to drafts")

                

                with col3:

                    if st.button("🎯 Annotate", key=f"annotate_{i}"):

                        # Convert to annotation task

                        switch_to_annotation_mode(edited, draft)

                

                with col4:

                    # Quality feedback

                    rating = st.feedback("stars", key=f"rate_{i}")

                    if rating is not None:

                        log_suggestion_quality(draft, rating)



def generate_tweet_suggestions(user_id: str, intent: str, context: str, 

                              n: int, max_length: int, archetypes: List[str],

                              stats: Dict) -> List[Dict]:

    """Generate tweet suggestions based on user's annotations"""

    

    # Mock generation based on user's annotation patterns

    base_patterns = [a for a in st.session_state.annotations if a["subgame"] in ["product", "marketing"]]

    

    suggestions = []

    for i in range(n):

        # Sample a pattern

        pattern = random.choice(base_patterns) if base_patterns else {

            "action_taken": "simplify_product",

            "state_before": {"friction": 0.7}

        }

        

        # Generate tweet based on pattern

        if pattern["subgame"] == "product":

            tweet = f"Spent {random.randint(1, 4)} weeks {pattern['action_taken'].replace('_', ' ')}. Result: {random.randint(20, 100)}% improvement in activation."

        else:

            tweet = f"Key insight: {pattern['action_taken'].replace('_', ' ')}. Took {random.randint(1, 12)} months to validate."

        

        suggestions.append({

            "tweet": tweet[:max_length*5],  # Rough word count

            "archetype": random.choice(archetypes),

            "confidence": random.uniform(0.6, 0.9),

            "based_on_pattern": pattern["action_taken"]

        })

    

    return suggestions



def switch_to_annotation_mode(tweet: str, draft: Optional[Dict] = None):

    """Switch to annotation tab with pre-filled tweet"""

    st.session_state.current_page = "gym"

    st.session_state.pending_annotation = tweet

    st.session_state.draft_metadata = draft

    st.success("Switched to annotation mode!")



def log_suggestion_quality(draft: Dict, rating: int):

    """Log user rating for future training"""

    quality_entry = {

        "draft": draft,

        "rating": rating,

        "timestamp": datetime.now().isoformat()

    }

    

    if "suggestion_feedback" not in st.session_state:

        st.session_state.suggestion_feedback = []

    

    st.session_state.suggestion_feedback.append(quality_entry)



# ============================================================================

# PROGRESS DASHBOARD

# ============================================================================



def progress_dashboard():

    st.title("📊 Your Decision Twin Dashboard")

    

    # Overall stats

    col1, col2, col3, col4 = st.columns(4)

    

    with col1:

        st.metric("Total Annotations", len(st.session_state.annotations))

    

    with col2:

        st.metric("Total XP", st.session_state.xp)

    

    with col3:

        current = LEVELS[st.session_state.current_level]

        st.metric("Current Level", current.name)

    

    with col4:

        if st.session_state.model_trained:

            st.metric("Model Status", "✅ Trained")

        else:

            st.metric("Model Status", "⏳ Not trained")

    

    # Subgame distribution

    st.subheader("Your Strategic Focus")

    if st.session_state.annotations:

        df = pd.DataFrame(st.session_state.annotations)

        subgame_counts = df["subgame"].value_counts()

        

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(subgame_counts)

        

        with col2:

            st.dataframe(subgame_counts)

    else:

        st.info("No annotations yet. Start in the Gym tab!")

    

    # Annotation timeline

    st.subheader("Annotation Progress")

    if st.session_state.annotations:

        # Mock timeline data

        dates = [datetime.now().timestamp() - (i * 86400) for i in range(len(st.session_state.annotations))]

        progress_df = pd.DataFrame({

            "date": dates,

            "cumulative": list(range(1, len(st.session_state.annotations) + 1))

        })

        

        st.line_chart(progress_df.set_index("date"))

    

    # Model performance (if trained)

    if st.session_state.model_trained and "suggestion_feedback" in st.session_state:

        st.subheader("Model Performance")

        feedback_df = pd.DataFrame(st.session_state.suggestion_feedback)

        

        if not feedback_df.empty:

            avg_rating = feedback_df["rating"].mean()

            st.metric("Average Suggestion Rating", f"{avg_rating:.1f} ⭐")

            

            # Rating distribution

            st.bar_chart(feedback_df["rating"].value_counts().sort_index())

    

    # Export and sharing

    st.subheader("Community & Export")

    col1, col2 = st.columns(2)

    

    with col1:

        st.download_button("📥 Export All Data (JSON)", 

                          json.dumps(st.session_state.annotations, indent=2),

                          "complete_dataset.json")

    

    with col2:

        if st.button("🌐 Submit to Community Leaderboard"):

            st.info("Feature coming soon: Share anonymized patterns with community")



# ============================================================================

# MAIN APP

# ============================================================================



def main():

    st.set_page_config(

        page_title="Startup MDP Trainer",

        page_icon="🎯",

        layout="wide",

        initial_sidebar_state="expanded"

    )

    

    # Initialize session state

    init_session_state()

    

    # Sidebar navigation

    st.sidebar.title("🎯 Navigation")

    

    pages = {

        "onboarding": "🏠 Onboarding",

        "tutorial": "🎓 Tutorial",

        "gym": "🏋️ Annotation Gym",

        "training": "🧬 Model Training",

        "assistant": "✍️ Writing Assistant",

        "dashboard": "📊 Dashboard"

    }

    

    # Set current page if not set

    if "current_page" not in st.session_state:

        st.session_state.current_page = "onboarding"

    

    # Sidebar buttons

    for page_key, page_label in pages.items():

        if st.sidebar.button(page_label, use_container_width=True):

            st.session_state.current_page = page_key

    

    st.sidebar.markdown("---")

    st.sidebar.info("""

    **Quick Stats**:

    - Annotations: {}

    - XP: {}

    - Level: {}

    """.format(

        len(st.session_state.annotations),

        st.session_state.xp,

        LEVELS[st.session_state.current_level].name

    ))

    

    # Page routing

    page = st.session_state.current_page

    

    if page == "onboarding":

        onboarding_screen()

    elif page == "tutorial":

        tutorial_screen()

    elif page == "gym":

        annotation_gym()

    elif page == "training":

        model_training_screen()

    elif page == "assistant":

        writing_assistant_screen()

    elif page == "dashboard":

        progress_dashboard()

    else:

        st.error("Page not found")



if __name__ == "__main__":

    main()

