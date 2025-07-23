"""
Monitoring Dashboard for AI Agent Performance and Guardrails
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime, timedelta
import json

# Import our services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.services.monitoring import MonitoringService, AgentType, OperationStatus
    from src.services.guardrails import AIGuardrails
    from src.services.feedback_system import FeedbackSystem
    from src.services.agent_personas import AgentPersonalizationService, PersonaMode
    from src.services.voice_service import VoiceService
    from src.services.cot_visualizer import CoTVisualizer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize services (with caching)
@st.cache_resource
def init_services():
    monitoring = MonitoringService()
    guardrails = AIGuardrails()
    feedback_system = FeedbackSystem(monitoring)
    personas = AgentPersonalizationService()
    voice_service = VoiceService()
    cot_visualizer = CoTVisualizer()
    
    return monitoring, guardrails, feedback_system, personas, voice_service, cot_visualizer

def main():
    st.title("🔍 AI Agent Monitoring Dashboard")
    st.markdown("Real-time monitoring of AI agent performance, guardrails, and user feedback")
    
    # Initialize services
    monitoring, guardrails, feedback_system, personas, voice_service, cot_visualizer = init_services()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    # Time range selection
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
        index=2
    )
    
    # Agent filter
    agent_filter = st.sidebar.multiselect(
        "Filter by Agent Type",
        [agent.value for agent in AgentType],
        default=[agent.value for agent in AgentType]
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Real-time Metrics", 
        "🛡️ Guardrails", 
        "💬 Feedback Analytics", 
        "🎭 Agent Personas",
        "🎤 Voice Commands",
        "🧠 Reasoning Chains"
    ])
    
    with tab1:
        show_realtime_metrics(monitoring, time_range, agent_filter)
    
    with tab2:
        show_guardrails_dashboard(guardrails, monitoring)
    
    with tab3:
        show_feedback_analytics(feedback_system)
    
    with tab4:
        show_agent_personas(personas)
    
    with tab5:
        show_voice_dashboard(voice_service)
    
    with tab6:
        show_cot_visualization(cot_visualizer)

def show_realtime_metrics(monitoring: MonitoringService, time_range: str, agent_filter: list):
    """Display real-time metrics dashboard"""
    st.header("📊 Real-time Performance Metrics")
    
    # Get metrics data
    try:
        real_time_data = monitoring.get_real_time_metrics()
        
        # Convert time range to hours
        hours_map = {
            "Last Hour": 1,
            "Last 6 Hours": 6, 
            "Last 24 Hours": 24,
            "Last Week": 168
        }
        hours = hours_map.get(time_range, 24)
        
        historical_data = monitoring.get_historical_metrics(hours=hours)
        
        # Overall metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_ops = real_time_data["overall_metrics"]["total_operations"]
            st.metric(
                "Total Operations",
                total_ops,
                delta=f"+{total_ops} in last hour"
            )
        
        with col2:
            success_rate = real_time_data["overall_metrics"]["success_rate"]
            st.metric(
                "Success Rate",
                f"{success_rate:.1%}",
                delta=f"{'↑' if success_rate > 0.9 else '↓'} {success_rate:.1%}"
            )
        
        with col3:
            avg_response = real_time_data["overall_metrics"]["avg_response_time"]
            st.metric(
                "Avg Response Time",
                f"{avg_response:.2f}s",
                delta=f"{'↓' if avg_response < 2.0 else '↑'} {avg_response:.2f}s"
            )
        
        with col4:
            total_tokens = real_time_data["overall_metrics"]["total_tokens"]
            st.metric(
                "Total Tokens",
                f"{total_tokens:,}",
                delta=f"+{total_tokens:,} tokens"
            )
        
        # Health status
        health_status = real_time_data["health_status"]
        status_colors = {
            "healthy": "🟢",
            "degraded": "🟡", 
            "unhealthy": "🔴",
            "unknown": "⚪"
        }
        
        st.info(f"System Health: {status_colors.get(health_status, '⚪')} {health_status.title()}")
        
        # Alerts
        alerts = real_time_data["alerts"]
        if alerts:
            st.warning("⚠️ Active Alerts:")
            for alert in alerts:
                st.error(f"**{alert['type']}**: {alert['message']}")
        
        # Agent-specific metrics
        st.subheader("Agent Performance Breakdown")
        
        agent_metrics = real_time_data["agent_metrics"]
        if agent_metrics:
            # Filter by selected agents
            filtered_metrics = {k: v for k, v in agent_metrics.items() if k in agent_filter}
            
            if filtered_metrics:
                # Create metrics DataFrame
                metrics_df = pd.DataFrame(filtered_metrics).T
                metrics_df.index.name = "Agent Type"
                
                # Display metrics table
                st.dataframe(
                    metrics_df.style.format({
                        'success_rate': '{:.1%}',
                        'avg_response_time': '{:.2f}s',
                        'total_tokens': '{:,}'
                    }),
                    use_container_width=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Success rate chart
                    fig_success = px.bar(
                        x=list(filtered_metrics.keys()),
                        y=[v["success_rate"] for v in filtered_metrics.values()],
                        title="Success Rate by Agent",
                        labels={"x": "Agent Type", "y": "Success Rate"}
                    )
                    fig_success.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_success, use_container_width=True)
                
                with col2:
                    # Response time chart
                    fig_response = px.bar(
                        x=list(filtered_metrics.keys()),
                        y=[v["avg_response_time"] for v in filtered_metrics.values()],
                        title="Average Response Time by Agent",
                        labels={"x": "Agent Type", "y": "Response Time (s)"}
                    )
                    st.plotly_chart(fig_response, use_container_width=True)
        
        # Historical trends
        if historical_data["hourly_breakdown"]:
            st.subheader("Historical Trends")
            
            # Convert to DataFrame
            hourly_df = pd.DataFrame(historical_data["hourly_breakdown"])
            hourly_df['hour'] = pd.to_datetime(hourly_df['hour'])
            
            # Filter by selected agents
            hourly_df = hourly_df[hourly_df['agent_type'].isin(agent_filter)]
            
            if not hourly_df.empty:
                # Operations over time
                fig_ops = px.line(
                    hourly_df,
                    x='hour',
                    y='operations',
                    color='agent_type',
                    title="Operations Over Time"
                )
                st.plotly_chart(fig_ops, use_container_width=True)
                
                # Response time trend
                fig_time = px.line(
                    hourly_df,
                    x='hour',
                    y='avg_response_time',
                    color='agent_type',
                    title="Response Time Trend"
                )
                st.plotly_chart(fig_time, use_container_width=True)
        
        # Error breakdown
        if historical_data["error_breakdown"]:
            st.subheader("Error Analysis")
            
            error_df = pd.DataFrame(historical_data["error_breakdown"])
            
            fig_errors = px.bar(
                error_df,
                x='count',
                y='error',
                orientation='h',
                title="Top Errors"
            )
            st.plotly_chart(fig_errors, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

def show_guardrails_dashboard(guardrails: AIGuardrails, monitoring: MonitoringService):
    """Display guardrails and security dashboard"""
    st.header("🛡️ AI Guardrails & Security")
    
    # Guardrails statistics
    stats = guardrails.get_guardrail_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rate Limited Users", stats["rate_limit_tracker_users"])
    
    with col2:
        st.metric("Sensitive Patterns", stats["sensitive_patterns_count"])
    
    with col3:
        st.metric("Malicious Patterns", stats["malicious_patterns_count"])
    
    with col4:
        st.metric("Prohibited Topics", stats["prohibited_topics_count"])
    
    # Test guardrails
    st.subheader("🧪 Test Guardrails")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Validation Test**")
        test_input = st.text_area("Enter test input:", value="")
        
        if st.button("Test Input Guardrails"):
            if test_input:
                result = guardrails.validate_input(test_input, "test_user")
                
                if result.passed:
                    st.success(f"✅ {result.message}")
                else:
                    st.error(f"❌ {result.message}")
                    st.warning(f"Threat Level: {result.threat_level.value}")
                
                # Show metadata
                if result.metadata:
                    st.json(result.metadata)
    
    with col2:
        st.write("**Output Validation Test**")
        test_output = st.text_area("Enter test output:", value="")
        
        if st.button("Test Output Guardrails"):
            if test_output:
                result = guardrails.validate_output(test_output)
                
                if result.passed:
                    st.success(f"✅ {result.message}")
                else:
                    st.error(f"❌ {result.message}")
                    st.warning(f"Threat Level: {result.threat_level.value}")
                
                # Show sanitized output
                sanitized = guardrails.sanitize_output(test_output)
                if sanitized != test_output:
                    st.write("**Sanitized Output:**")
                    st.code(sanitized)
    
    # Guardrail violations (simulated data for demo)
    st.subheader("Recent Guardrail Events")
    
    # Create sample data
    sample_events = [
        {"timestamp": datetime.now() - timedelta(hours=1), "type": "Rate Limit", "severity": "Medium", "user": "user_123"},
        {"timestamp": datetime.now() - timedelta(hours=2), "type": "Sensitive Data", "severity": "High", "user": "user_456"},
        {"timestamp": datetime.now() - timedelta(hours=3), "type": "Malicious Content", "severity": "Critical", "user": "user_789"},
    ]
    
    events_df = pd.DataFrame(sample_events)
    st.dataframe(events_df, use_container_width=True)

def show_feedback_analytics(feedback_system: FeedbackSystem):
    """Display feedback analytics dashboard"""
    st.header("💬 User Feedback Analytics")
    
    try:
        # Get feedback dashboard data
        feedback_data = feedback_system.get_feedback_dashboard_data()
        
        # Overall feedback metrics
        overall = feedback_data["overall_stats"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Feedback", overall["total_feedback"])
        
        with col2:
            st.metric("Average Rating", f"{overall['avg_rating']}/5.0")
        
        with col3:
            st.metric("Helpfulness Rate", f"{overall['helpfulness_rate']:.1%}")
        
        with col4:
            st.metric("Corrections Provided", overall["corrections_provided"])
        
        # Agent feedback breakdown
        if feedback_data["agent_feedback"]:
            st.subheader("Feedback by Agent Type")
            
            agent_df = pd.DataFrame(feedback_data["agent_feedback"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rating = px.bar(
                    agent_df,
                    x='agent_type',
                    y='avg_rating',
                    title="Average Rating by Agent"
                )
                st.plotly_chart(fig_rating, use_container_width=True)
            
            with col2:
                fig_helpful = px.bar(
                    agent_df,
                    x='agent_type',
                    y='helpfulness_rate',
                    title="Helpfulness Rate by Agent"
                )
                fig_helpful.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig_helpful, use_container_width=True)
        
        # Recent insights
        if feedback_data["recent_insights"]:
            st.subheader("Recent Improvement Insights")
            
            for insight in feedback_data["recent_insights"]:
                priority_colors = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }
                
                priority_icon = priority_colors.get(insight["priority"], "⚪")
                
                st.info(f"{priority_icon} **{insight['pattern_type']}** (Frequency: {insight['frequency']})\n{insight['description']}")
        
        # Top improvement areas
        if feedback_data["top_improvement_areas"]:
            st.subheader("Top Areas for Improvement")
            
            improvement_df = pd.DataFrame(feedback_data["top_improvement_areas"])
            
            fig_improvement = px.bar(
                improvement_df,
                x='issue_count',
                y='operation',
                color='agent_type',
                orientation='h',
                title="Issues by Operation"
            )
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        # Feedback collection interface
        st.subheader("💭 Provide Feedback")
        
        with st.form("feedback_form"):
            agent_type = st.selectbox("Agent Type", [agent.value for agent in AgentType])
            operation = st.text_input("Operation", value="test_operation")
            user_input = st.text_area("Your Input")
            ai_output = st.text_area("AI Response")
            rating = st.slider("Rating", 1, 5, 3)
            helpful = st.checkbox("Was this helpful?")
            feedback_text = st.text_area("Additional Feedback")
            correct_answer = st.text_area("Correct Answer (if different)")
            
            if st.form_submit_button("Submit Feedback"):
                if user_input and ai_output:
                    feedback_id = feedback_system.record_feedback(
                        agent_type=AgentType(agent_type),
                        operation=operation,
                        user_input=user_input,
                        ai_output=ai_output,
                        user_id="dashboard_user",
                        rating=rating,
                        helpful=helpful,
                        feedback_text=feedback_text,
                        correct_answer=correct_answer
                    )
                    
                    st.success(f"Feedback recorded! ID: {feedback_id}")
                else:
                    st.error("Please provide both user input and AI output")
    
    except Exception as e:
        st.error(f"Error loading feedback data: {str(e)}")

def show_agent_personas(personas: AgentPersonalizationService):
    """Display agent personalization dashboard"""
    st.header("🎭 Agent Personas & Personalization")
    
    # Available personas
    available_personas = personas.get_available_personas()
    
    st.subheader("Available Persona Modes")
    
    for persona in available_personas:
        with st.expander(f"🎭 {persona['name']} Mode"):
            st.write(f"**Description:** {persona['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Characteristics:**")
                chars = persona['characteristics']
                st.write(f"- Tone: {chars['tone']}")
                st.write(f"- Verbosity: {chars['verbosity']}")
                st.write(f"- Expertise Level: {chars['expertise_level']}")
                st.write(f"- Max Length: {chars['max_length']} chars")
            
            with col2:
                st.write("**Features:**")
                st.write(f"- Examples: {'✅' if chars['includes_examples'] else '❌'}")
                st.write(f"- Explanations: {'✅' if chars['includes_explanations'] else '❌'}")
                st.write(f"- Technical Terms: {'✅' if chars['uses_technical_terms'] else '❌'}")
                st.write(f"- Format: {chars['response_format']}")
    
    # Persona configuration
    st.subheader("🛠️ Configure Your Persona")
    
    with st.form("persona_config"):
        user_id = st.text_input("User ID", value="dashboard_user")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.selectbox("Mode", [mode.value for mode in PersonaMode])
            tone = st.selectbox("Tone", ["formal", "casual", "friendly", "authoritative", "concise"])
            verbosity = st.selectbox("Verbosity", ["concise", "normal", "detailed"])
            response_format = st.selectbox("Response Format", ["bullet_points", "paragraphs", "structured"])
        
        with col2:
            include_examples = st.checkbox("Include Examples", value=True)
            include_explanations = st.checkbox("Include Explanations", value=True)
            use_technical_terms = st.checkbox("Use Technical Terms", value=False)
            max_length = st.slider("Max Response Length", 100, 3000, 1000)
        
        if st.form_submit_button("Save Persona Configuration"):
            updates = {
                "mode": PersonaMode(mode),
                "tone": tone,
                "verbosity": verbosity,
                "include_examples": include_examples,
                "include_explanations": include_explanations,
                "use_technical_terms": use_technical_terms,
                "response_format": response_format,
                "max_response_length": max_length
            }
            
            updated_config = personas.update_user_persona(user_id, updates)
            st.success("Persona configuration saved!")
            
            # Show generated system prompt
            st.subheader("Generated System Prompt")
            system_prompt = personas.generate_system_prompt(updated_config, "treasury")
            st.code(system_prompt, language="text")

def show_voice_dashboard(voice_service: VoiceService):
    """Display voice command dashboard"""
    st.header("🎤 Voice Commands Dashboard")
    
    # Voice service status
    status = voice_service.get_voice_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Service Enabled", "✅" if status["enabled"] else "❌")
    
    with col2:
        st.metric("Dependencies Available", "✅" if status["available"] else "❌")
    
    with col3:
        st.metric("Microphone Ready", "✅" if status.get("microphone_available", False) else "❌")
    
    if not status["available"]:
        st.warning("Voice dependencies not installed. Run: `pip install SpeechRecognition pyttsx3 pyaudio`")
        return
    
    # Voice settings
    st.subheader("🔧 Voice Settings")
    
    if status["enabled"]:
        settings_data = status["settings"]
        
        with st.form("voice_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox("Language", ["en-US", "en-GB", "es-ES", "fr-FR"], 
                                      index=0 if settings_data["language"] == "en-US" else 0)
                speech_rate = st.slider("Speech Rate (WPM)", 50, 300, settings_data["speech_rate"])
                volume = st.slider("Volume", 0.0, 1.0, settings_data["volume"])
            
            with col2:
                recognition_timeout = st.slider("Recognition Timeout (s)", 1.0, 30.0, settings_data["recognition_timeout"])
                phrase_timeout = st.slider("Phrase Timeout (s)", 0.5, 10.0, settings_data["phrase_timeout"])
            
            if st.form_submit_button("Update Voice Settings"):
                new_settings = {
                    "language": language,
                    "speech_rate": speech_rate,
                    "volume": volume,
                    "recognition_timeout": recognition_timeout,
                    "phrase_timeout": phrase_timeout
                }
                
                success = voice_service.set_voice_settings(new_settings)
                if success:
                    st.success("Voice settings updated!")
                else:
                    st.error("Failed to update voice settings")
    
    # Available voices
    if status["enabled"]:
        st.subheader("🗣️ Available Voices")
        
        voices = voice_service.get_available_voices()
        if voices:
            voices_df = pd.DataFrame(voices)
            st.dataframe(voices_df, use_container_width=True)
        else:
            st.info("No voice information available")
    
    # Voice testing
    st.subheader("🎯 Test Voice Commands")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Text-to-Speech Test**")
        test_text = st.text_area("Enter text to speak:", value="Hello, this is a test of the treasury AI voice system.")
        
        if st.button("🔊 Speak Text"):
            if status["enabled"] and test_text:
                result = voice_service.speak_text(test_text)
                if result["success"]:
                    st.success("✅ Text spoken successfully!")
                    st.info(f"Estimated duration: {result.get('duration_estimate', 0):.1f} seconds")
                else:
                    st.error(f"❌ Error: {result['error']}")
            else:
                st.warning("Voice service not available or no text provided")
    
    with col2:
        st.write("**Speech-to-Text Test**")
        
        if st.button("🎤 Listen for Speech"):
            if status["enabled"]:
                with st.spinner("Listening... Speak now!"):
                    result = voice_service.listen_for_speech(timeout=5.0)
                
                if result["success"]:
                    st.success(f"✅ Recognized: {result['text']}")
                    st.info(f"Language: {result.get('language', 'unknown')}")
                else:
                    st.error(f"❌ Error: {result['error']}")
                    
                    if result.get("timeout"):
                        st.warning("No speech detected within timeout period")
                    elif result.get("unclear"):
                        st.warning("Speech was unclear - try speaking more clearly")
            else:
                st.warning("Voice service not available")

def show_cot_visualization(cot_visualizer: CoTVisualizer):
    """Display chain-of-thought visualization dashboard"""
    st.header("🧠 Chain-of-Thought Reasoning")
    
    # Recent chains
    recent_chains = cot_visualizer.get_recent_chains(limit=10)
    
    if recent_chains:
        st.subheader("Recent Reasoning Chains")
        
        # Convert to DataFrame for display
        chains_df = pd.DataFrame(recent_chains)
        chains_df['timestamp'] = pd.to_datetime(chains_df['timestamp'], unit='s')
        
        # Display table
        st.dataframe(
            chains_df[['query', 'agent_type', 'operation', 'success', 'step_count', 'confidence', 'duration']].style.format({
                'confidence': '{:.2f}',
                'duration': '{:.2f}s'
            }),
            use_container_width=True
        )
        
        # Select chain for detailed view
        st.subheader("Detailed Chain Analysis")
        
        selected_chain_id = st.selectbox(
            "Select Chain for Detailed View",
            options=[chain['chain_id'] for chain in recent_chains],
            format_func=lambda x: f"{recent_chains[next(i for i, c in enumerate(recent_chains) if c['chain_id'] == x)]['query'][:50]}..."
        )
        
        if selected_chain_id:
            # Get visualization data
            viz_data = cot_visualizer.generate_visualization_data(selected_chain_id)
            
            if "error" not in viz_data:
                # Chain summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Steps", viz_data["metadata"]["step_count"])
                
                with col2:
                    st.metric("Completed Steps", viz_data["metadata"]["completed_steps"])
                
                with col3:
                    st.metric("Success", "✅" if viz_data["success"] else "❌")
                
                with col4:
                    confidence = viz_data["total_confidence"]
                    st.metric("Confidence", f"{confidence:.2f}" if confidence else "N/A")
                
                # Query and final output
                st.write(f"**Query:** {viz_data['query']}")
                if viz_data["final_output"]:
                    st.write(f"**Final Output:** {viz_data['final_output'][:200]}...")
                
                # Reasoning steps
                st.subheader("Reasoning Steps")
                
                for i, node in enumerate(viz_data["nodes"]):
                    with st.expander(f"Step {i+1}: {node['title']} ({node['status']})"):
                        st.write(f"**Type:** {node['type']}")
                        st.write(f"**Description:** {node['description']}")
                        
                        if node['reasoning']:
                            st.write(f"**Reasoning:** {node['reasoning']}")
                        
                        if node['confidence']:
                            st.write(f"**Confidence:** {node['confidence']:.2f}")
                        
                        if node['duration']:
                            st.write(f"**Duration:** {node['duration']:.2f}s")
                        
                        if node['input_data']:
                            st.write("**Input Data:**")
                            st.json(node['input_data'])
                        
                        if node['output_data']:
                            st.write("**Output Data:**")
                            st.json(node['output_data'])
                        
                        if node['alternatives']:
                            st.write(f"**Alternatives Considered:** {', '.join(node['alternatives'])}")
                
                # Mermaid diagram
                st.subheader("Reasoning Flow Diagram")
                
                mermaid_diagram = cot_visualizer.generate_mermaid_diagram(selected_chain_id)
                
                # Display using streamlit-mermaid if available, otherwise show code
                try:
                    from streamlit_mermaid import st_mermaid
                    st_mermaid(mermaid_diagram)
                except ImportError:
                    st.code(mermaid_diagram, language="mermaid")
                    st.info("Install streamlit-mermaid for interactive diagram: `pip install streamlit-mermaid`")
            
            else:
                st.error(viz_data["error"])
    
    else:
        st.info("No reasoning chains available. Chains will appear here as agents process queries.")
        
        # Demo chain creation
        st.subheader("Create Demo Chain")
        
        if st.button("Generate Demo Reasoning Chain"):
            # Create a sample reasoning chain for demonstration
            chain_id = cot_visualizer.start_reasoning_chain(
                query="Analyze transaction for journal mapping",
                agent_type="journal_mapper",
                operation="analyze_transaction"
            )
            
            # Add sample steps
            from src.services.cot_visualizer import ThoughtType, StepStatus
            
            step1 = cot_visualizer.add_reasoning_step(
                chain_id=chain_id,
                thought_type=ThoughtType.INPUT_ANALYSIS,
                title="Parse Transaction Data",
                description="Extract transaction type, amount, and asset class",
                input_data={"transaction": "BUY BONDS 1000000 USD"},
                reasoning="Identified as a purchase transaction for bonds"
            )
            
            cot_visualizer.complete_reasoning_step(
                chain_id=chain_id,
                step_id=step1,
                output_data={"type": "BUY", "asset": "BONDS", "amount": 1000000, "currency": "USD"},
                confidence=0.95
            )
            
            step2 = cot_visualizer.add_reasoning_step(
                chain_id=chain_id,
                thought_type=ThoughtType.KNOWLEDGE_RETRIEVAL,
                title="Retrieve Journal Rules",
                description="Find matching journal mapping rules",
                reasoning="Searching for rules matching bond purchase transactions"
            )
            
            cot_visualizer.complete_reasoning_step(
                chain_id=chain_id,
                step_id=step2,
                output_data={"rules_found": 3, "best_match": "bond_purchase_rule"},
                confidence=0.88
            )
            
            step3 = cot_visualizer.add_reasoning_step(
                chain_id=chain_id,
                thought_type=ThoughtType.DECISION_POINT,
                title="Select Account Mapping",
                description="Choose debit and credit accounts",
                reasoning="Bond purchase requires debit to investment account, credit to cash"
            )
            
            cot_visualizer.complete_reasoning_step(
                chain_id=chain_id,
                step_id=step3,
                output_data={"debit_account": "Investment Assets", "credit_account": "Cash - USD"},
                confidence=0.92
            )
            
            cot_visualizer.finish_reasoning_chain(
                chain_id=chain_id,
                final_output="Journal Entry: Debit Investment Assets $1,000,000, Credit Cash - USD $1,000,000",
                success=True
            )
            
            st.success("Demo reasoning chain created! Refresh to see it in the list.")
            st.rerun()

if __name__ == "__main__":
    main() 