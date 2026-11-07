import yaml
from typing import Dict, Optional, Any

class FSMEngine:
    def __init__(self, config_path: str):
        self.states: Dict[str, Dict[str, Any]] = {}
        self.current_state: Optional[str] = None
        self.load_config(config_path)
    
    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        for state_name, state_data in config.items():
            next_states = {}
            for k, v in state_data.get('next_states', {}).items():
                next_states[str(k)] = str(v)
            
            self.states[state_name] = {
                'name': state_name,
                'description': state_data.get('description', ''),
                'prompt': state_data.get('prompt', ''),
                'next_states': next_states,
                'is_final': bool(state_data.get('is_final', False))
            }
    
    def start(self, initial_state: str = 'danger'):
        if initial_state not in self.states:
            raise ValueError(f"Initial state '{initial_state}' not found")
        self.current_state = initial_state
        return self.get_current_prompt()
    
    def get_current_prompt(self) -> Optional[str]:
        if self.current_state is None:
            return None
        return self.states[self.current_state]['prompt']
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        if self.current_state is None:
            return None
        return self.states[self.current_state]
    
    def transition(self, user_input: str) -> Optional[str]:
        if self.current_state is None:
            return None
        
        current_state_obj = self.states[self.current_state]
        
        for keyword, next_state in current_state_obj['next_states'].items():
            if keyword in user_input:
                self.current_state = next_state
                return self.get_current_prompt()
        
        return None
    
    def is_final(self) -> bool:
        if self.current_state is None:
            return True
        return self.states[self.current_state]['is_final']
    
    def reset(self):
        self.current_state = None
