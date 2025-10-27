"""Retrieval Behaviour for SPADE agents."""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from ..rag.retrievers import BaseRetriever
from ..utils.retrieval_utils import create_retrieval_response_body

logger = logging.getLogger("spade_llm.behaviour")


class RetrievalBehaviour(CyclicBehaviour):
    """
    A specialized behaviour for SPADE agents that handles document retrieval.

    This behaviour extends the CyclicBehaviour to:
    - Receive XMPP messages with retrieval queries
    - Process queries using configured retrievers
    - Return relevant documents to requesters
    - Support various retrieval parameters (k, filters, scores)
    - Track retrieval statistics and performance

    The behaviour remains active throughout the agent's lifecycle,
    continuously processing incoming retrieval requests.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reply_to: Optional[str] = None,
        default_k: int = 4,
        include_scores: bool = False,
        on_retrieval_complete: Optional[Callable[[str, List[Any]], None]] = None,
    ):
        """
        Initialize the retrieval behaviour.

        Args:
            retriever: The retriever to use for document search
            reply_to: JID to send responses to. If None, replies to the original sender
            default_k: Default number of documents to retrieve
            include_scores: Whether to include similarity scores in responses
            on_retrieval_complete: Callback function when retrieval completes
                                  (receives query and results)
        """
        super().__init__()
        self.retriever = retriever
        self.reply_to = reply_to
        self.default_k = default_k
        self.include_scores = include_scores
        self.on_retrieval_complete = on_retrieval_complete

        # Track processed messages to avoid duplicates
        self._processed_messages: Set[Union[str, int]] = set()

        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_documents_retrieved": 0,
            "average_retrieval_time": 0.0,
            "last_query_time": None,
        }

    async def run(self):
        """
        Main execution loop for the behaviour.
        Waits for messages containing retrieval queries, processes them,
        and sends back relevant documents.
        """
        # Wait for incoming message
        msg = await self.receive(timeout=10)

        if not msg:
            return

        # Check if we've already processed this message
        if msg.id in self._processed_messages:
            logger.debug(f"Skipping already processed message {msg.id}")
            return

        # Mark message as processed
        self._processed_messages.add(msg.id)
        logger.debug(f"RetrievalBehaviour received message: {msg}")

        # Extract query and parameters from message
        try:
            query_data = self._parse_query_message(msg)
            query = query_data.get("query")
            k = query_data.get("k", self.default_k)
            filters = query_data.get("filters")
            include_scores = query_data.get("include_scores", self.include_scores)
            search_type = query_data.get("search_type", "similarity")

            if not query:
                await self._send_error_response(
                    msg, "No query found in message. Expected 'query' field."
                )
                return

            logger.info(
                f"Processing retrieval query: '{query}' (k={k}, search_type={search_type})"
            )

            # Perform retrieval
            start_time = time.time()
            results = await self._perform_retrieval(
                query, k, filters, include_scores, search_type
            )
            retrieval_time = time.time() - start_time

            # Update statistics
            self._update_stats(results, retrieval_time)

            # Callback
            if self.on_retrieval_complete:
                self.on_retrieval_complete(query, results)

            # Send response
            await self._send_retrieval_response(
                msg, query, results, retrieval_time, include_scores
            )

        except Exception as e:
            logger.error(f"Error processing retrieval request: {e}", exc_info=True)
            self._stats["failed_retrievals"] += 1
            await self._send_error_response(msg, str(e))

    def _parse_query_message(self, msg: Message) -> Dict[str, Any]:
        """
        Parse query parameters from message body.

        Supports both simple string queries and JSON-formatted queries with parameters.

        Args:
            msg: The incoming message

        Returns:
            Dictionary with query parameters
        """
        body = msg.body.strip() if msg.body else ""

        # Try parsing as JSON first
        try:
            query_data = json.loads(body)
            if isinstance(query_data, dict):
                return query_data
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback to treating entire body as query string
        return {"query": body}

    async def _perform_retrieval(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        include_scores: bool,
        search_type: str = "similarity",
    ) -> List[Any]:
        """
        Perform the actual retrieval operation.

        Args:
            query: The search query
            k: Number of documents to retrieve
            filters: Optional metadata filters
            include_scores: Whether to include similarity scores
            search_type: Type of search ("similarity", "similarity_score", "mmr")

        Returns:
            List of documents or (document, score) tuples
        """
        kwargs = {}
        if filters:
            kwargs["filters"] = filters

        # Determine search type based on include_scores if not explicitly set
        if search_type == "similarity" and include_scores:
            search_type = "similarity_score"

        # Use the unified retrieve method with search_type
        results = await self.retriever.retrieve(
            query, 
            k, 
            search_type=search_type,
            **kwargs
        )

        return results

    async def _send_retrieval_response(
        self,
        original_msg: Message,
        query: str,
        results: List[Any],
        retrieval_time: float,
        include_scores: bool,
    ):
        """
        Send retrieval results back to the requester.

        The response contains only the documents found. Metadata like
        retrieval time and result count are included in message headers
        for observability but not in the body.

        Args:
            original_msg: The original query message
            query: The query string (for metadata)
            results: Retrieved documents or (document, score) tuples
            retrieval_time: Time taken for retrieval (for metadata)
            include_scores: Whether results include scores
        """
        # Determine recipient
        recipient = self.reply_to or str(original_msg.sender)

        # Create reply message (documents)
        reply = Message(to=recipient)
        reply.body = create_retrieval_response_body(results, include_scores)
        reply.thread = original_msg.thread
        
        # Metadata for observability
        reply.set_metadata("message_type", "retrieval_response")
        reply.set_metadata("query", query)
        reply.set_metadata("num_results", str(len(results)))
        reply.set_metadata("retrieval_time", str(round(retrieval_time, 3)))

        logger.info(f"Sending {len(results)} results to {recipient}")
        
        try:
            await self.send(reply)
            logger.info(f"Retrieval response sent successfully to {recipient}")
        except Exception as e:
            logger.error(f"Error sending retrieval response to {recipient}: {e}")

    async def _send_error_response(self, original_msg: Message, error_message: str):
        """
        Send error response to the requester.

        Args:
            original_msg: The original query message
            error_message: The error message to send
        """
        recipient = self.reply_to or str(original_msg.sender)

        error_data = {"error": error_message, "query": original_msg.body}

        reply = Message(to=recipient)
        reply.body = json.dumps(error_data, indent=2)
        reply.thread = original_msg.thread
        reply.set_metadata("message_type", "retrieval_error")

        logger.info(f"Sending error response to {recipient}")

        try:
            await self.send(reply)
        except Exception as e:
            logger.error(f"Error sending error response to {recipient}: {e}")

    def _update_stats(self, results: List[Any], retrieval_time: float):
        """
        Update retrieval statistics.

        Args:
            results: Retrieved results
            retrieval_time: Time taken for retrieval
        """
        self._stats["total_queries"] += 1
        self._stats["successful_retrievals"] += 1
        self._stats["total_documents_retrieved"] += len(results)
        self._stats["last_query_time"] = time.time()

        # Update rolling average for retrieval time
        total = self._stats["total_queries"]
        current_avg = self._stats["average_retrieval_time"]
        self._stats["average_retrieval_time"] = (
            current_avg * (total - 1) + retrieval_time
        ) / total

    def update_retriever(self, new_retriever: BaseRetriever) -> None:
        """
        Update the retriever used by this behaviour.

        Args:
            new_retriever: The new retriever to use
        """
        self.retriever = new_retriever
        logger.info("Retriever updated in behaviour")

    def set_default_k(self, k: int) -> None:
        """
        Set the default number of documents to retrieve.

        Args:
            k: Number of documents to retrieve by default
        """
        self.default_k = k
        logger.info(f"Default k set to {k}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.

        Returns:
            Dictionary with retrieval statistics
        """
        return self._stats.copy()