import ollama
import time
from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedCoT:
    def __init__(self, model_name: str = "llama3.2-vision:11b-instruct-q4_K_M", num_branches: int = 3, 
                 temperature_range: tuple = (0.1, 0.8)):
        self.model = model_name
        self.num_branches = num_branches
        self.temp_range = temperature_range
        self.performance_history = []
        
    def generate_thought_branches(self, prompt: str) -> List[str]:
        """Generate multiple thought branches using different perspectives"""
        thoughts = []
        temperatures = np.linspace(self.temp_range[0], self.temp_range[1], self.num_branches)
        
        def get_branch(temp):
            response = ollama.generate(model=self.model,
                                     prompt=f"Think step by step:\n{prompt}",
                                     temperature=temp)
            return response['response']
            
        with ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(get_branch, temperatures))
        return thoughts
    
    def verify_consistency(self, thoughts: List[str]) -> Dict[str, float]:
        """Verify consistency between different thought branches"""
        consistency_scores = {}
        for i, thought in enumerate(thoughts):
            # Calculate semantic similarity with other thoughts
            score = ollama.generate(
                model=self.model,
                prompt=f"Rate the logical consistency of this thought from 0-1:\n{thought}")
            consistency_scores[f"branch_{i}"] = float(score['response'])
        return consistency_scores
    
    def optimize_response(self, thoughts: List[str], 
                         consistency_scores: Dict[str, float]) -> str:
        """Select the most reliable thought branch"""
        best_branch_idx = max(consistency_scores.items(), 
                            key=lambda x: x[1])[0].split('_')[1]
        return thoughts[int(best_branch_idx)]
    
    def monitor_performance(self, execution_time: float, 
                          consistency_score: float) -> None:
        """Monitor and track performance metrics"""
        self.performance_history.append({
            'execution_time': execution_time,
            'consistency_score': consistency_score
        })
        
    def dynamic_temperature_adjustment(self) -> None:
        """Adjust temperature range based on performance history"""
        if len(self.performance_history) > 5:
            recent_scores = [p['consistency_score'] 
                           for p in self.performance_history[-5:]]
            if np.mean(recent_scores) < 0.5:
                self.temp_range = (max(0.1, self.temp_range[0] - 0.1),
                                 max(0.3, self.temp_range[1] - 0.1))
                
    def enhanced_cot(self, prompt: str) -> str:
        """Main method implementing the enhanced CoT process"""
        start_time = time.time()
        
        try:
            # Generate multiple thought branches
            thoughts = self.generate_thought_branches(prompt)
            
            # Verify consistency
            consistency_scores = self.verify_consistency(thoughts)
            
            # Get optimized response
            final_response = self.optimize_response(thoughts, consistency_scores)
            
            # Monitor performance
            execution_time = time.time() - start_time
            avg_consistency = np.mean(list(consistency_scores.values()))
            self.monitor_performance(execution_time, avg_consistency)
            
            # Adjust parameters dynamically
            self.dynamic_temperature_adjustment()
            
            return final_response
            
        except Exception as e:
            # Fallback to simple completion
            return ollama.generate(model=self.model, 
                                 prompt=prompt)['response']

# Usage example
if __name__ == "__main__":
    enhanced_cot = EnhancedCoT()
    result = enhanced_cot.enhanced_cot("Explain the process of photosynthesis")
    print(result)