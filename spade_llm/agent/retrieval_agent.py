"""Retrieval agent implementation for SPADE."""

import logging
from typing import Any, Callable, Dict, List, Optional

from spade.agent import Agent
from spade.template import Template

from ..behaviour import RetrievalBehaviour
from ..rag.retrievers import BaseRetriever

logger = logging.getLogger("spade_llm.agent")


class RetrievalAgent(Agent):
    """
    A SPADE agent specialized in document retrieval operations.

    This agent extends the standard SPADE agent with:
    - Built-in retriever integration
    - SPADE message-based query processing
    - Integration with RAG (Retrieval-Augmented Generation) components
    
    The RetrievalAgent can work standalone or as part of a MAS,
    communicating with SPADE agents.
    """

    def __init__(
        self,
        jid: str,
        password: str,
        retriever: BaseRetriever,
        reply_to: Optional[str] = None,
        default_k: int = 4,
        include_scores: bool = False,
        on_retrieval_complete: Optional[Callable[[str, List[Any]], None]] = None,
        verify_security: bool = False
    ):
        """
        Initialize agent.

        Args:
            jid: The Jabber ID of the agent
            password: The password for the agent
            retriever: The retriever to use for document search
            reply_to: JID to send responses to. If None, replies to the original sender
            default_k: Default number of documents to retrieve
            include_scores: Whether to include similarity scores in responses
            on_retrieval_complete: Callback function when retrieval completes 
                                  (receives query and results)
            verify_security: Whether to verify security certificates
        """
        super().__init__(jid, password, verify_security=verify_security)
        
        self.retriever = retriever
        self.reply_to = reply_to
        self.default_k = default_k
        self.include_scores = include_scores
        self.on_retrieval_complete = on_retrieval_complete

        self.retrieval_behaviour = RetrievalBehaviour(
            retriever=self.retriever,
            reply_to=self.reply_to,
            default_k=self.default_k,
            include_scores=self.include_scores,
            on_retrieval_complete=self.on_retrieval_complete,
        )

    async def setup(self):
        """Set up the agent with retrieval behaviour."""
        logger.info(f"RetrievalAgent starting: {self.jid}")

        # Retrieval-targeted messages only
        template = Template()
        template.set_metadata("message_type", "retrieval")
        self.add_behaviour(self.retrieval_behaviour, template)

    def update_retriever(self, new_retriever: BaseRetriever) -> None:
        """
        Update the retriever used by this agent.

        Args:
            new_retriever: The new retriever to use
        """
        self.retriever = new_retriever
        self.retrieval_behaviour.update_retriever(new_retriever)
        logger.info(f"Retriever updated for agent {self.jid}")

    def set_default_k(self, k: int) -> None:
        """
        Set the default number of documents to retrieve.

        Args:
            k: Number of documents to retrieve by default
        """
        self.default_k = k
        self.retrieval_behaviour.set_default_k(k)
        logger.info(f"Default k set to {k} for agent {self.jid}")

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics for this agent.

        Returns:
            Dictionary with retrieval statistics
        """
        return self.retrieval_behaviour.get_stats()
