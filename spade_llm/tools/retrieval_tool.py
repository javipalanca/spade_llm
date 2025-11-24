"""Retrieval tool for LLM agents to query Retrieval agents."""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template
from .llm_tool import LLMTool

if TYPE_CHECKING:
    from ..agent.llm_agent import LLMAgent

logger = logging.getLogger("spade_llm.tools")


class RetrievalTool(LLMTool):
    """A tool that enables LLM agents to query Retrieval Agents for documents."""

    class _WaitForResponseBehaviour(OneShotBehaviour):
        """Private behaviour for handling retrieval responses."""
        
        def __init__(self, query_msg: Message, response_timeout: int):
            """
            Initialize the retrieval response behaviour.
            
            Args:
                query_msg: The query message to send
                response_timeout: Timeout for waiting for response
            """
            super().__init__()
            self.response: Optional[Message] = None
            self.query_msg = query_msg
            self.response_timeout = response_timeout
        
        async def run(self):
            """Send query and wait for response."""
            # Send the query
            await self.send(self.query_msg)
            
            # Wait for response
            response = await self.receive(timeout=self.response_timeout)
            if response:
                self.response = response

    def __init__(
        self,
        retrieval_agent_jid: str,
        agent_instance: Optional["LLMAgent"] = None,
        default_k: int = 4,
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
            timeout: Maximum time to wait for retrieval response (seconds)
            name: Name of the tool (default: "retrieve_documents")
            description: Custom description (uses default if None)
        """
        self.retrieval_agent_jid = retrieval_agent_jid
        self.agent_instance = agent_instance
        self.default_k = default_k
        self.timeout = timeout

        # Validate parameters to ensure consistency with tool schema
        if not 1 <= self.default_k <= 20:
            raise ValueError(f"default_k must be between 1 and 20, but got {self.default_k}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be a positive integer, but got {self.timeout}")

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

    def set_agent(self, agent_instance: "LLMAgent") -> None:
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

        # Validate query is not empty or None
        if not query or (isinstance(query, str) and not query.strip()):
            error_msg = "Query cannot be empty, None, or contain only whitespace"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        # Validate k parameter if provided
        k_value = k if k is not None else self.default_k
        if not 1 <= k_value <= 20:
            error_msg = f"Parameter 'k' must be between 1 and 20, but got {k_value}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            # Prepare query message
            query_data = {
                "query": query,
                "k": k_value,
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
        if self.agent_instance is None:
            raise ValueError("Agent instance is not set. Please call 'set_agent' to initialize the agent instance.")
        
        # Create template to match retrieval responses
        template = Template()
        template.sender = str(self.retrieval_agent_jid)
        template.set_metadata("message_type", "retrieval_response")
        
        response_behaviour = self._WaitForResponseBehaviour(msg, self.timeout)
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
