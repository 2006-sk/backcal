"""
ChromaDB service for storing and retrieving embeddings.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.core.config import settings


class ChromaService:
    def __init__(self):
        self.use_chroma = settings.USE_CHROMA
        self.api_key = settings.CHROMA_API_KEY
        self.tenant = settings.CHROMA_TENANT
        self.database = settings.CHROMA_DATABASE
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        
        if not self.use_chroma:
            print("[Chroma] ChromaDB disabled. Set USE_CHROMA=true to enable.")
            self.client = None
            self.collection = None
            return
        
        if not self.api_key or not self.tenant or not self.database:
            print("[Chroma] Missing Cloud credentials. Set CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE")
            self.client = None
            self.collection = None
            return
        
        try:
            print(f"[Chroma] Connecting to Cloud: tenant={self.tenant}, database={self.database}")
            # Initialize ChromaDB Cloud client
            self.client = chromadb.CloudClient(
                api_key=self.api_key,
                tenant=self.tenant,
                database=self.database
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "AdAtlas AI embeddings"}
            )
            
            print(f"[Chroma] Connected to Cloud collection: {self.collection_name}")
            
        except Exception as e:
            print(f"[Chroma] Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            self.client = None
            self.collection = None
    
    def enabled(self) -> bool:
        """Check if ChromaDB is enabled and working."""
        return self.use_chroma and self.client is not None and self.collection is not None
    
    def store_embeddings(
        self,
        file_id: str,
        file_name: str,
        file_type: str,
        audio_embedding: List[float],
        visual_embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store audio and visual embeddings in ChromaDB.
        
        Args:
            file_id: Unique file identifier
            file_name: Original filename
            file_type: 'video' or 'image'
            audio_embedding: Audio embedding vector
            visual_embedding: Visual embedding vector
            metadata: Additional metadata (duration, scene_count, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled():
            print("[Chroma] ChromaDB not enabled, skipping storage")
            return False
        
        try:
            # Store audio embedding
            self.collection.add(
                embeddings=[audio_embedding] if audio_embedding else [[0.0] * 768],
                ids=[f"{file_id}_audio"],
                metadatas=[{
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "embedding_type": "audio",
                    **metadata
                }]
            )
            
            # Store visual embedding
            self.collection.add(
                embeddings=[visual_embedding] if visual_embedding else [[0.0] * 768],
                ids=[f"{file_id}_visual"],
                metadatas=[{
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "embedding_type": "visual",
                    **metadata
                }]
            )
            
            print(f"[Chroma] Stored embeddings for {file_id}")
            return True
            
        except Exception as e:
            print(f"[Chroma] Failed to store embeddings: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        embedding_type: str = "visual",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content using embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embedding_type: 'audio' or 'visual'
            n_results: Number of results to return
            
        Returns:
            List of similar content with metadata
        """
        if not self.enabled():
            print("[Chroma] ChromaDB not enabled")
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"embedding_type": embedding_type}
            )
            
            # Format results
            formatted_results = []
            if results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "id": doc_id,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else None
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"[Chroma] Search failed: {e}")
            return []
    
    def delete_embeddings(self, file_id: str) -> bool:
        """
        Delete embeddings for a specific file.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if successful
        """
        if not self.enabled():
            return False
        
        try:
            self.collection.delete(ids=[f"{file_id}_audio", f"{file_id}_visual"])
            print(f"[Chroma] Deleted embeddings for {file_id}")
            return True
        except Exception as e:
            print(f"[Chroma] Failed to delete: {e}")
            return False


# Global instance
chroma_service = ChromaService()

