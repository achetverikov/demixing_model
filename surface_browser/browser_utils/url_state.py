"""
URL State Management
===================

Simple URL state management using query parameters with gzip compression.
"""

import streamlit as st
import json
import gzip
import base64
from typing import Dict, Any, Optional


class URLStateManager:
    """Manages application state persistence using compressed query parameters."""
    
    def _compress_state(self, state_dict: Dict[str, Any]) -> str:
        """Compress state dictionary to URL-safe string."""
        try:
            # Convert to compact JSON
            json_str = json.dumps(state_dict, separators=(',', ':'))
            # Compress with gzip
            compressed = gzip.compress(json_str.encode('utf-8'))
            # Base64 encode for URL safety
            encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
            return encoded
        except Exception:
            return ""
    
    def _decompress_state(self, encoded_state: str) -> Optional[Dict[str, Any]]:
        """Decompress state string back to dictionary."""
        try:
            # Base64 decode
            compressed = base64.urlsafe_b64decode(encoded_state.encode('ascii'))
            # Decompress
            json_str = gzip.decompress(compressed).decode('utf-8')
            # Parse JSON
            return json.loads(json_str)
        except Exception:
            return None
    
    def save_state(self, state_dict: Dict[str, Any]):
        """Save state to URL query parameters with compression."""
        try:
            # Always store in session state as backup
            st.session_state["app_state"] = state_dict
            
            # Compress state
            compressed_state = self._compress_state(state_dict)
            
            if compressed_state:
                # Clear existing query params and set new state
                st.query_params.clear()
                st.query_params["s"] = compressed_state
            
        except Exception:
            # Fallback to session state only
            st.session_state["app_state"] = state_dict
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from URL query parameters or session state."""
        try:
            # First try to get from URL query parameters
            compressed_state = st.query_params.get("s")
            if compressed_state:
                state = self._decompress_state(compressed_state)
                if state:
                    # Update session state with loaded state
                    st.session_state["app_state"] = state
                    return state
        except Exception:
            pass
        
        # Fallback to session state
        return st.session_state.get("app_state")
    
    def get_state_with_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Get current state with fallback to defaults."""
        saved_state = self.load_state()
        if saved_state:
            result = defaults.copy()
            result.update(saved_state)
            return result
        return defaults
    
    def update_state(self, updates: Dict[str, Any]):
        """Update specific parts of the state and save to URL."""
        current_state = self.load_state() or {}
        current_state.update(updates)
        self.save_state(current_state)


# Global state manager instance
state_manager = URLStateManager()


def save_app_state(**state_updates):
    """Convenience function to save app state."""
    state_manager.update_state(state_updates)


def load_app_state(defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to load app state with defaults."""
    if defaults is None:
        defaults = {}
    return state_manager.get_state_with_defaults(defaults)


def init_url_state_js():
    """No-op function for compatibility."""
    pass