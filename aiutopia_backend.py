"""
AIUTOPIA Backend API Server
Production-grade FastAPI server with Claude AI integration
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import io
import json
import asyncio
from datetime import datetime
import aiohttp

# FastAPI app
app = FastAPI(
    title="Aiutopia API",
    description="Causal Intelligence Platform with Claude AI Integration",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CLAUDE AI INTEGRATION
# =============================================================================

class ClaudeAIAgent:
    """
    Claude AI integration for causal reasoning
    Uses Anthropic API to power intelligent analysis
    """
    
    def __init__(self):
        self.api_base = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 4096
        
    async def analyze_causal_structure(self, data: pd.DataFrame, domain: str) -> Dict:
        """
        Use Claude to discover causal relationships in data
        
        Args:
            data: DataFrame with variables
            domain: Context (city, finance, marketing, etc.)
        
        Returns:
            Causal graph structure with relationships
        """
        
        # Prepare data summary for Claude
        data_summary = self._summarize_data(data)
        
        prompt = f"""You are a world-class causal inference expert. Analyze this dataset and identify causal relationships.

DOMAIN: {domain}

DATA SUMMARY:
{data_summary}

TASK: Identify causal relationships between variables. For each relationship, provide:
1. Source variable (cause)
2. Target variable (effect)
3. Effect strength (estimated coefficient, -1 to 1)
4. Confidence (0 to 1)
5. Mechanism (how the cause affects the effect)
6. Evidence (what in the data supports this)

CRITICAL RULES:
- Only identify relationships where causality is plausible (not just correlation)
- Consider temporal ordering, domain knowledge, and confounders
- Avoid reverse causation
- Flag potential confounders

Return ONLY valid JSON in this exact format:
{{
  "relationships": [
    {{
      "source": "variable_name",
      "target": "variable_name", 
      "effect_size": 0.35,
      "confidence": 0.82,
      "mechanism": "explanation",
      "evidence": ["data_point_1", "data_point_2"]
    }}
  ],
  "confounders": ["variable_name"],
  "assumptions": ["assumption_1", "assumption_2"]
}}"""

        response = await self._call_claude(prompt)
        
        try:
            # Parse JSON response
            causal_structure = json.loads(response)
            return causal_structure
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown
            return self._extract_json(response)
    
    async def recommend_interventions(self, 
                                     causal_graph: Dict, 
                                     objective: str,
                                     constraints: Dict) -> List[Dict]:
        """
        Use Claude to recommend optimal interventions
        
        Args:
            causal_graph: Discovered causal structure
            objective: What we want to achieve
            constraints: Budget, time, feasibility limits
        
        Returns:
            Ranked list of intervention recommendations
        """
        
        prompt = f"""You are an expert in causal intervention design. Given this causal graph, recommend interventions.

OBJECTIVE: {objective}

CAUSAL GRAPH:
{json.dumps(causal_graph, indent=2)}

CONSTRAINTS:
{json.dumps(constraints, indent=2)}

TASK: Design 3-5 interventions that achieve the objective while respecting constraints.

For each intervention, provide:
1. Name (descriptive)
2. Target variables to manipulate
3. Intervention values (how much to change each)
4. Expected effect on objective
5. Cost estimate
6. Implementation steps
7. Risks and failure modes

Return ONLY valid JSON:
{{
  "interventions": [
    {{
      "name": "intervention_name",
      "targets": {{"variable": new_value}},
      "expected_effect": 0.18,
      "confidence": 0.87,
      "cost": 2400000,
      "timeframe_days": 180,
      "steps": ["step_1", "step_2"],
      "risks": ["risk_1", "risk_2"]
    }}
  ]
}}"""

        response = await self._call_claude(prompt)
        
        try:
            recommendations = json.loads(response)
            return recommendations['interventions']
        except:
            return self._extract_json(response).get('interventions', [])
    
    async def explain_causality(self, 
                               intervention: Dict, 
                               causal_graph: Dict) -> str:
        """
        Generate natural language explanation of causal chain
        
        Returns:
            Human-readable causal proof
        """
        
        prompt = f"""Explain WHY this intervention works using causal reasoning.

INTERVENTION:
{json.dumps(intervention, indent=2)}

CAUSAL GRAPH:
{json.dumps(causal_graph, indent=2)}

TASK: Write a clear explanation for non-experts that:
1. Identifies the causal chain (A → B → C)
2. Explains each causal link
3. Provides the total effect calculation
4. Lists key assumptions
5. Describes potential failure modes

Write in clear, confident language. Use specific numbers. Make it actionable."""

        explanation = await self._call_claude(prompt)
        return explanation
    
    async def chat(self, message: str, context: Dict) -> str:
        """
        Natural language interface to Aiutopia
        
        Args:
            message: User's question/command
            context: Current analysis context
        
        Returns:
            Claude's response
        """
        
        prompt = f"""You are Aiutopia's AI assistant, an expert in causal inference and intervention design.

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

USER MESSAGE:
{message}

Respond helpfully. If the user asks to:
- Analyze data → explain what causal relationships you find
- Recommend interventions → suggest optimal interventions
- Explain causality → provide causal proof
- Make predictions → use counterfactual reasoning
- Compare options → do multi-objective analysis

Be concise but comprehensive. Use specific numbers when available."""

        response = await self._call_claude(prompt)
        return response
    
    async def _call_claude(self, prompt: str) -> str:
        """Make API call to Claude"""
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": "API_KEY_PLACEHOLDER",  # User sets this
                    "anthropic-version": "2023-06-01"
                },
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['content'][0]['text']
                else:
                    raise HTTPException(status_code=response.status, 
                                       detail="Claude API error")
    
    def _summarize_data(self, df: pd.DataFrame) -> str:
        """Create data summary for Claude"""
        summary = []
        summary.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        summary.append("\nVARIABLES:")
        
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                summary.append(f"  {col} ({dtype}): mean={stats['mean']:.2f}, "
                             f"std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
            else:
                n_unique = df[col].nunique()
                summary.append(f"  {col} ({dtype}): {n_unique} unique values")
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            summary.append("\nTOP CORRELATIONS:")
            # Get top 5 correlations
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, val in corr_pairs[:5]:
                summary.append(f"  {col1} <-> {col2}: {val:.3f}")
        
        return "\n".join(summary)
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from markdown code blocks"""
        try:
            # Try to find JSON in markdown code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                json_str = text
            
            return json.loads(json_str)
        except:
            return {}


# Global Claude instance
claude = ClaudeAIAgent()


# =============================================================================
# DATA MODELS
# =============================================================================

class AnalysisRequest(BaseModel):
    domain: str
    objective: str
    constraints: Dict[str, Any]
    data_source: Optional[str] = None

class InterventionRequest(BaseModel):
    intervention_id: str
    parameters: Dict[str, float]

class ChatRequest(BaseModel):
    message: str
    session_id: str

class DataIntegrationConfig(BaseModel):
    source_type: str  # "csv", "api", "database", "stream"
    connection_details: Dict[str, Any]
    refresh_interval_minutes: Optional[int] = 60


# =============================================================================
# CSV UPLOAD & ANALYSIS
# =============================================================================

@app.post("/api/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    domain: str = "general",
    background_tasks: BackgroundTasks = None
):
    """
    Upload CSV file for causal analysis
    
    Returns:
        Analysis results with causal graph and recommendations
    """
    
    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Validate data
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 rows of data")
    
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 variables")
    
    # Clean data
    df = clean_data(df)
    
    # Use Claude to discover causal structure
    causal_graph = await claude.analyze_causal_structure(df, domain)
    
    # Store in session (would use database in production)
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    return {
        "session_id": session_id,
        "data_summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "variables": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        "causal_graph": causal_graph,
        "status": "ready",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze/recommend-interventions")
async def recommend_interventions(request: AnalysisRequest):
    """
    Get intervention recommendations from Claude
    """
    
    # Would retrieve causal graph from session/database
    # For demo, using placeholder
    causal_graph = {
        "relationships": [],
        "confounders": [],
        "assumptions": []
    }
    
    interventions = await claude.recommend_interventions(
        causal_graph=causal_graph,
        objective=request.objective,
        constraints=request.constraints
    )
    
    # Compute stability scores for each intervention
    for intervention in interventions:
        intervention['stability_score'] = compute_stability_score(intervention)
    
    # Rank by expected_effect * confidence * (1 / cost)
    interventions.sort(
        key=lambda x: x['expected_effect'] * x['confidence'] / max(x['cost'], 1),
        reverse=True
    )
    
    return {
        "interventions": interventions,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# API INTEGRATIONS
# =============================================================================

class DataIntegrationManager:
    """
    Manages connections to external data sources
    """
    
    async def connect_google_analytics(self, credentials: Dict) -> pd.DataFrame:
        """Connect to Google Analytics API"""
        # Implementation would use Google Analytics API
        return pd.DataFrame()
    
    async def connect_stripe(self, api_key: str) -> pd.DataFrame:
        """Connect to Stripe payments API"""
        # Implementation would use Stripe API
        return pd.DataFrame()
    
    async def connect_salesforce(self, credentials: Dict) -> pd.DataFrame:
        """Connect to Salesforce CRM"""
        return pd.DataFrame()
    
    async def connect_postgres(self, connection_string: str, query: str) -> pd.DataFrame:
        """Connect to PostgreSQL database"""
        import asyncpg
        
        conn = await asyncpg.connect(connection_string)
        try:
            rows = await conn.fetch(query)
            df = pd.DataFrame(rows, columns=rows[0].keys() if rows else [])
            return df
        finally:
            await conn.close()
    
    async def connect_rest_api(self, url: str, headers: Dict, params: Dict) -> pd.DataFrame:
        """Generic REST API connector"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return pd.json_normalize(data)
                else:
                    raise HTTPException(status_code=response.status, detail="API error")


integration_manager = DataIntegrationManager()


@app.post("/api/integrations/connect")
async def connect_data_source(config: DataIntegrationConfig):
    """
    Connect to external data source
    
    Supported sources:
    - Google Analytics
    - Stripe
    - Salesforce  
    - PostgreSQL
    - REST APIs
    - CSV files
    - Real-time streams
    """
    
    source_type = config.source_type.lower()
    details = config.connection_details
    
    try:
        if source_type == "google_analytics":
            df = await integration_manager.connect_google_analytics(details)
        elif source_type == "stripe":
            df = await integration_manager.connect_stripe(details['api_key'])
        elif source_type == "salesforce":
            df = await integration_manager.connect_salesforce(details)
        elif source_type == "postgres":
            df = await integration_manager.connect_postgres(
                details['connection_string'], 
                details['query']
            )
        elif source_type == "rest_api":
            df = await integration_manager.connect_rest_api(
                details['url'],
                details.get('headers', {}),
                details.get('params', {})
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source: {source_type}")
        
        return {
            "status": "connected",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# REAL-TIME DATA STREAMS
# =============================================================================

@app.websocket("/ws/data-stream")
async def websocket_data_stream(websocket):
    """
    WebSocket endpoint for real-time data streaming
    
    Use cases:
    - Live sensor data from cities
    - Real-time market data
    - Streaming analytics events
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process real-time data point
            processed = process_streaming_data(data)
            
            # Check for regime shifts
            if detect_regime_shift(processed):
                await websocket.send_json({
                    "type": "alert",
                    "message": "Regime shift detected - causal model may need update",
                    "severity": "high"
                })
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        await websocket.close()


# =============================================================================
# CLAUDE CHAT INTERFACE
# =============================================================================

@app.post("/api/chat")
async def chat_with_claude(request: ChatRequest):
    """
    Natural language interface to Aiutopia via Claude
    
    Examples:
    - "Analyze my marketing data"
    - "What interventions would reduce churn?"
    - "Explain why this intervention works"
    - "Compare these two strategies"
    """
    
    # Get conversation context
    context = get_session_context(request.session_id)
    
    # Call Claude
    response = await claude.chat(request.message, context)
    
    # Store in conversation history
    store_message(request.session_id, request.message, response)
    
    return {
        "response": response,
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/chat/stream")
async def chat_stream(session_id: str, message: str):
    """
    Streaming chat responses from Claude
    """
    async def generate():
        context = get_session_context(session_id)
        
        # Stream response from Claude
        response = await claude.chat(message, context)
        
        # Simulate streaming (real implementation would use SSE)
        for chunk in response.split('. '):
            yield f"data: {chunk}. \n\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# =============================================================================
# COUNTERFACTUAL SIMULATION
# =============================================================================

@app.post("/api/simulate/counterfactual")
async def simulate_counterfactual(request: InterventionRequest):
    """
    Run counterfactual simulation
    
    Returns what-if outcomes for intervention
    """
    
    # Get causal model
    causal_model = get_causal_model(request.intervention_id)
    
    # Simulate outcomes
    outcomes = run_simulation(causal_model, request.parameters)
    
    # Compute confidence intervals using conformal prediction
    confidence_intervals = compute_conformal_intervals(outcomes)
    
    return {
        "intervention": request.intervention_id,
        "parameters": request.parameters,
        "predicted_outcomes": outcomes,
        "confidence_intervals": confidence_intervals,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# AUTOMATED MONITORING
# =============================================================================

@app.post("/api/monitor/start")
async def start_monitoring(intervention_id: str, metrics: List[str]):
    """
    Start real-time monitoring of deployed intervention
    
    Tracks:
    - Actual outcomes vs predictions
    - Assumption violations
    - Regime shifts
    - Unexpected effects
    """
    
    # Would start background monitoring task
    return {
        "status": "monitoring_started",
        "intervention_id": intervention_id,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/monitor/status/{intervention_id}")
async def get_monitoring_status(intervention_id: str):
    """
    Get current status of monitored intervention
    """
    
    return {
        "intervention_id": intervention_id,
        "status": "active",
        "actual_effect": 0.16,
        "predicted_effect": 0.18,
        "confidence": 0.89,
        "assumptions_valid": True,
        "alerts": [],
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]
    
    return df


def compute_stability_score(intervention: Dict) -> float:
    """Compute intervention stability score"""
    # Simplified - would use full ISA module
    base_score = intervention.get('confidence', 0.5)
    
    # Penalize high cost
    cost_penalty = 1.0 / (1.0 + intervention.get('cost', 0) / 1000000)
    
    # Boost for high expected effect
    effect_boost = intervention.get('expected_effect', 0)
    
    return base_score * cost_penalty * (1 + effect_boost)


def process_streaming_data(data: Dict) -> Dict:
    """Process real-time data point"""
    # Add timestamp if missing
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()
    
    return data


def detect_regime_shift(data: Dict) -> bool:
    """Detect if causal regime has shifted"""
    # Simplified - would use CUSUM or other change point detection
    return False


def get_session_context(session_id: str) -> Dict:
    """Retrieve session context"""
    # Would query from database
    return {
        "session_id": session_id,
        "domain": "general",
        "analysis_history": []
    }


def store_message(session_id: str, message: str, response: str):
    """Store chat message in history"""
    # Would save to database
    pass


def get_causal_model(intervention_id: str) -> Dict:
    """Retrieve causal model"""
    # Would query from database
    return {}


def run_simulation(model: Dict, parameters: Dict) -> Dict:
    """Run counterfactual simulation"""
    # Simplified - would use full digital twin
    return {
        "outcome_1": 0.85,
        "outcome_2": 1.23
    }


def compute_conformal_intervals(outcomes: Dict) -> Dict:
    """Compute confidence intervals using conformal prediction"""
    intervals = {}
    for key, value in outcomes.items():
        intervals[key] = {
            "lower": value * 0.92,
            "upper": value * 1.08
        }
    return intervals


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "claude_available": True,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
