"""Retrieval tool for LLM agents to query Retrieval agents."""

import json
import logging
from typing import Any, Dict, Optional

from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template

from .llm_tool import LLMTool

logger = logging.getLogger("spade_llm.tools")


class RetrievalTool(LLMTool):
    """A tool that enables LLM agents to query Retrieval Agents for documents."""

    def __init__(
        self,
        retrieval_agent_jid: str,
        agent_instance: Optional[Any] = None,
        default_k: int = 4,
        include_scores: bool = False,
        timeout: int = 30,
        name: str = "retrieve_documents",
        description: Optional[str] = None,
    ):
        """
        Initialize the retrieval tool.

        Args:
            retrieval_agent_jid: JID of the Retrieval Agent to query
            agent_instance: The LLM agent instance (will be set later if None)
            default_k: Default number of documents to retrieve
            include_scores: Whether to request similarity scores
            timeout: Maximum time to wait for retrieval response (seconds)
            name: Name of the tool (default: "retrieve_documents")
            description: Custom description (uses default if None)
        """
        self.retrieval_agent_jid = retrieval_agent_jid
        self.agent_instance = agent_instance
        self.default_k = default_k
        self.include_scores = include_scores
        self.timeout = timeout

        if description is None:
            description = """Retrieve relevant documents from the knowledge base by querying the Retrieval Agent.
            
            Use this tool to:
            - Find information needed to answer user questions
            - Get context about specific topics
            
            The retrieval agent will return the most relevant documents matching your query.
            Results include document content and metadata."""

        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific about what information you need.",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": default_k,
                        "description": f"Number of documents to retrieve (default: {default_k})",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters to narrow search",
                    },
                },
                "required": ["query"],
            },
            func=self._retrieve_documents,
        )

    def set_agent(self, agent_instance: Any) -> None:
        """
        Set the agent instance for sending messages.

        Args:
            agent_instance: The LLM agent that will use this tool
        """
        self.agent_instance = agent_instance
        logger.info(f"RetrievalTool bound to agent {agent_instance.jid}")

    async def _retrieve_documents(
        self, query: str, k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Retrieve documents from the Retrieval Agent.

        Args:
            query: The search query
            k: Number of documents to retrieve (uses default if None)
            filters: Optional metadata filters

        Returns:
            JSON string with retrieval results or error message
        """
        if not self.agent_instance:
            error_msg = "RetrievalTool not properly initialized with agent instance"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            # Prepare query message
            query_data = {
                "query": query,
                "k": k if k is not None else self.default_k,
                "include_scores": self.include_scores,
            }
            
            if filters:
                query_data["filters"] = filters

            msg = Message(to=self.retrieval_agent_jid)
            msg.body = json.dumps(query_data)
            msg.set_metadata("message_type", "retrieval")

            logger.info(f"Sending retrieval query to {self.retrieval_agent_jid}: '{query}'")

            # Send message and wait for response
            response = await self._send_and_wait_for_response(msg)

            if response:
                logger.info(f"Received retrieval response from {self.retrieval_agent_jid}")
                return self._format_response(response.body if response.body is not None else "")
            else:
                error_msg = f"Timeout waiting for response from Retrieval Agent (waited {self.timeout}s)"
                logger.warning(error_msg)
                return json.dumps({"error": error_msg, "query": query})

        except Exception as e:
            error_msg = f"Error during document retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg, "query": query})

    async def _send_and_wait_for_response(self, msg: Message) -> Optional[Message]:
        """
        Send a message and wait for a response.

        Args:
            msg: The message to send

        Returns:
            The response message or None if timeout
        """
        timeout = self.timeout
        
        # Create a one-shot behaviour to handle the retrieval response
        class RetrievalResponseBehaviour(OneShotBehaviour):
            def __init__(self, query_msg, response_timeout):
                super().__init__()
                self.response = None
                self.query_msg = query_msg
                self.response_timeout = response_timeout
            
            async def run(self):
                # Send the query
                await self.send(self.query_msg)
                
                # Wait for response
                response = await self.receive(timeout=self.response_timeout)
                if response:
                    self.response = response
        
        # Create template to match retrieval responses
        template = Template()
        template.sender = str(self.retrieval_agent_jid)
        template.set_metadata("message_type", "retrieval_response")
        
        # Create and add the behaviour
        response_behaviour = RetrievalResponseBehaviour(msg, timeout)
        
        if self.agent_instance is None:
            raise ValueError("Agent instance is not set. Please call 'set_agent' to initialize the agent instance.")
        self.agent_instance.add_behaviour(response_behaviour, template)
        
        # Wait for behaviour to complete
        await response_behaviour.join()
        
        return response_behaviour.response

    def _format_response(self, response_body: str) -> str:
        """
        Format the retrieval response for the LLM.

        Args:
            response_body: The raw response body from Retrieval Agent

        Returns:
            Formatted response string
        """
        try:
            data = json.loads(response_body)

            # Check for error
            if "error" in data:
                return json.dumps(data)

            # Extract documents
            documents = data.get("documents", [])

            if len(documents) == 0:
                return json.dumps({
                    "message": "No documents found matching your query.",
                })

            # Add rank for clarity
            for i, doc in enumerate(documents, 1):
                doc["rank"] = i

            return json.dumps({"documents": documents}, indent=2)

        except json.JSONDecodeError:
            logger.error(f"Failed to parse retrieval response: {response_body}")
            return json.dumps({
                "error": "Failed to parse retrieval response",
                "raw_response": response_body,
            })
