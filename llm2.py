import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import ollama
import json
from pathlib import Path

class CoT:
    def __init__(self, 
                 model_name: str, 
                 temperature: float = 0.7,
                 history_file: str = r'C:\Users\harvi\Codebases\LLMOps\CoT\performance_history.json'):
        self.model = model_name
        self.temperature = temperature
        self.history_file = Path(history_file)
        self._ensure_history_file()
    
    def _ensure_history_file(self) -> None:
        """Create history file if it doesn't exist"""
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text('[]')

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to JSON file"""
        try:
            data = json.loads(self.history_file.read_text()) if self.history_file.exists() else []
            data.append(metrics)
            self.history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")

    def analyze_complexity(self, prompt: str) -> Tuple[int, List[str]]:
        """Analyze query complexity and determine steps"""
        complexity_query = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'user', 
                'content': """Analyze the complexity of this query and:
                1. Determine number of steps needed (1-5)
                2. Break it down into clear steps
                Respond in format:
                STEPS: <number>
                1. First step
                2. Second step
                etc.
                Query: """ + prompt
            }]
        )
        
        response = complexity_query['message']['content']
        steps_count = int(response.split('STEPS:')[1].split('\n')[0].strip())
        steps = [s.strip() for s in response.split('\n') if s.strip() and any(str(i) in s for i in range(steps_count))]
        
        return steps_count, steps

    def generate_response(self, prompt: str, step_num: int, total_steps: int, previous_response: str = "") -> str:
        try:
            context = f"As per your plan of reponse, continue the response by refering to your previous steps:\n{previous_response}\n\nNow address step {step_num}/{total_steps} in NOT more than 100 words:" if previous_response else f"You have created a plan to respond more accurately and productively, As per the plan start elaborate the step {step_num}/{total_steps} in not more than 100 words:"
            
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': f"{context}\n{prompt}"}],
                stream=True,
                options={'temperature': self.temperature}
            )
            
            step_response = f"\nStep {step_num}/{total_steps}:\n"
            print(f"\n{step_response}", end='')
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_text = chunk['message']['content']
                    print(chunk_text, end='', flush=True)
                    step_response += chunk_text
            
            return step_response
        except Exception as e:
            print(f"Error in step {step_num}: {str(e)}")
            return ""

    def cot_main(self, prompt: str) -> Optional[str]:
        try:
            start_time = time.time()
            
            # Analyze complexity and get steps
            num_steps, stages = self.analyze_complexity(prompt)
            
            print(f"\nQuery requires {num_steps} steps:")
            for stage in stages:
                print(stage)
            
            # Generate responses for each stage
            full_response = ""
            for i, stage in enumerate(stages, 1):
                stage_resp = self.generate_response(stage, i, num_steps, full_response)
                full_response += f"{stage_resp}\n"
            
            # Calculate metrics
            end_time = time.time()
            metrics = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "time_taken": round(end_time - start_time, 2),
                "num_steps": num_steps,
                "total_tokens": len(full_response.split())
            }
            
            self._save_metrics(metrics)
            return full_response
            
        except Exception as e:
            print(f"Error in cot_main: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        cot = CoT(model_name="qwen2:7b")
        result = cot.cot_main("Explain quantum entanglement and its applications")
        if result:
            print("\nComplete Response:", result)
    except Exception as e:
        print(f"Error: {str(e)}")