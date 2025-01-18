import time
import numpy as np
from typing import List, Dict, Any, Optional
import ollama
import json
from pathlib import Path

class CoT:
    def __init__(self, 
                 model_name: str, 
                 num_branches: int = 3,
                 temperature_range: tuple = (0.1, 0.8),
                 history_file: str = r'C:\Users\harvi\Codebases\LLMOps\CoT\performance_history.json'):
        self.model = model_name
        self.num_branches = num_branches
        self.temp_range = temperature_range
        self.history_file = Path(history_file)
        self.ensure_history_file()
    
    def ensure_history_file(self) -> None:
        """Create history file if it doesn't exist"""
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text('[]')

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to JSON file"""
        try:
            if self.history_file.exists():
                data = json.loads(self.history_file.read_text())
            else:
                data = []
            
            data.append(metrics)
            self.history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")

    def generate_thought_branches(self, prompt: str) -> List[str]:
        """Generate multiple thought branches using different temperatures"""
        branches = []
        for _ in range(self.num_branches):
            try:
                temperature = np.random.uniform(*self.temp_range)
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True,
                    options={
                        'temperature': temperature
                    }
                )
                
                branch_text = ""
                print(f"\nBranch with temperature {temperature:.2f}:")
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        chunk_text = chunk['message']['content']
                        print(chunk_text, end='', flush=True)
                        branch_text += chunk_text
                print("\n")
                
                branches.append(branch_text)
            except Exception as e:
                print(f"Error generating branch: {str(e)}")
                continue
        
        return branches

    def cot_main(self, prompt: str) -> Optional[str]:
        """Main method implementing the enhanced CoT process"""
        try:
            start_time = time.time()
            branches = self.generate_thought_branches(prompt)
            
            if not branches:
                raise ValueError("No successful thought branches generated")
            
            final_response = "\n".join(branches)
            end_time = time.time()
            
            # Calculate metrics
            time_taken = end_time - start_time
            tokens = len(final_response.split())
            token_speed = tokens / time_taken if time_taken > 0 else 0
            
            metrics = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "time_taken": round(time_taken, 2),
                "token_speed": round(token_speed, 2),
                "num_branches": len(branches),
                "total_tokens": tokens
            }
            
            self.save_metrics(metrics)
            return final_response
            
        except Exception as e:
            print(f"Error in enhanced_cot: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        # Using a valid model name that's already pulled
        enhanced_cot = CoT(model_name="qwen2:7b")
        result = enhanced_cot.cot_main("Hi")
        if result:
            print("\nFinal combined response:", result)
    except Exception as e:
        print(f"Error: {str(e)}")