"""
Agent modules for Agentic Chaser.

8 specialized agents:
1. Client Communication Agent
2. Provider Communication Agent
3. Document Processing Agent
4. Sentiment Analysis Agent
5. Response Parser Agent
6. Workflow Orchestrator & Priority Agent (Core Brain)
7. Provider Portal RPA Agent
8. Predictive Intelligence Agent
"""

from agents.client_comms import client_communication_agent
from agents.document_processing import document_processing_agent
from agents.predictive_intelligence import predictive_intelligence_agent
from agents.provider_comms import provider_communication_agent
from agents.provider_rpa import provider_rpa_agent
from agents.response_parser import response_parser_agent
from agents.sentiment_analysis import sentiment_analysis_agent
from agents.workflow_orchestrator import workflow_orchestrator_agent

__all__ = [
    "client_communication_agent",
    "provider_communication_agent",
    "document_processing_agent",
    "sentiment_analysis_agent",
    "response_parser_agent",
    "workflow_orchestrator_agent",
    "provider_rpa_agent",
    "predictive_intelligence_agent",
]