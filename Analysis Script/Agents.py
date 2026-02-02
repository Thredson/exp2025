# -*- coding: utf-8 -*-
import json
import random
import datetime
from typing import Dict, List, Tuple
import os
import numpy as np

class RandomAgent:
    def __init__(self, subject_id: str = "random_agent_001", condition: str = "feature"):
        self.subject_id = subject_id
        self.condition = condition
        self.unchosen_counts = {}  # Track unchosen counts for reward calculation
        self.trial_counter = 0
        self.time_elapsed = 0
        
        # Initialize randomized images
        self.randomized_images = self._randomize_images()
        
        # All possible pairs
        self.all_pairs = [
            ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'),
            ('B', 'C'), ('B', 'D'), ('B', 'E'), ('B', 'F'),
            ('C', 'D'), ('C', 'E'), ('C', 'F'),
            ('D', 'E'), ('D', 'F'),
            ('E', 'F')
        ]
        
        self.data = []
        
    def _randomize_images(self) -> Dict[str, str]:
        """Randomize image mapping"""
        original_images = ['A', 'B', 'C', 'D', 'E', 'F']
        shuffled = original_images.copy()
        random.shuffle(shuffled)
        
        mapping = {}
        for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
            mapping[letter] = f"{self.condition}/image{shuffled[i]}.png"
        
        return mapping
    
    def _initialize_unchosen_counts(self):
        """Initialize unchosen counts for all images"""
        for img in ['A', 'B', 'C', 'D', 'E', 'F']:
            self.unchosen_counts[img] = 0
    
    def _calculate_reward(self, chosen_image: str, unchosen_image: str) -> bool:
        """Calculate reward based on the formula: P(reward) = 1 - (1-0.3)^(n+1)"""
        if not chosen_image:
            return False
        
        n_chosen = self.unchosen_counts.get(chosen_image, 0)
        n_unchosen = self.unchosen_counts.get(unchosen_image, 0)

        # Increment unchosen count
        self.unchosen_counts[unchosen_image] = n_unchosen + 1
        
        # Calculate reward probability
        reward_probability = 1 - (0.7 ** (n_chosen + 1))
        
        # Generate reward based on probability
        reward = random.random() < reward_probability

        # Reset chosen image count
        self.unchosen_counts[chosen_image] = 0
        
        return reward
    
    def make_choice(self, left_image: str, right_image: str) -> Tuple[str, str, str]:
        """Random agent chooses randomly between left and right"""
        if random.random() < 0.5:
            return left_image, right_image, 'left'
        else:
            return right_image, left_image, 'right'
    
    def run_trial(self, phase: str, block: int, pair: Tuple[str, str], 
                  trial_in_block: int) -> Dict:
        """Run a single trial"""
        self.trial_counter += 1
        
        # Randomly assign pair to left/right
        if random.random() < 0.5:
            left_image, right_image = pair
        else:
            right_image, left_image = pair
        
        # Make choice (random for this agent)
        chosen_image, unchosen_image, chosen_side = self.make_choice(left_image, right_image)
        
        # Calculate reward (only matters in training phase)
        reward = self._calculate_reward(chosen_image, unchosen_image)
        
        # Simulate time progression
        self.time_elapsed += random.randint(2000, 5000)
        
        # Build trial data
        trial_data = {
            "phase": phase,
            "block": block,
            "left_image": left_image,
            "actual_left_image": self.randomized_images[left_image],
            "right_image": right_image,
            "actual_right_image": self.randomized_images[right_image],
            "pair": f"{left_image}-{right_image}",
            "condition": self.condition,
            "stimulus": self._generate_stimulus(phase, left_image, right_image),
            "response": "f" if chosen_side == "left" else "j",
            "trial_type": "html-keyboard-response",
            "trial_index": len(self.data),
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed,
            "chosen_image": chosen_image,
            "unchosen_image": unchosen_image,
            "chosen_side": chosen_side,
            "reward": reward,
            "trial_number": self.trial_counter - 1
        }
        
        return trial_data
    
    def _generate_stimulus(self, phase: str, left_image: str, right_image: str) -> str:
        """Generate stimulus HTML string"""
        phase_text = "TRAINING PHASE" if phase == "training" else "TESTING PHASE"
        return f"""
                <div class="phase-indicator">{phase_text}</div>
                <div class="progress-info">Choose an image: F = Left, J = Right</div>
                <div class="image-pair-container">
                    <div class="image-choice">
                        <img src="{self.randomized_images[left_image]}" alt="Image {left_image}">
                        <div class="key-label">F</div>
                    </div>
                    <div class="image-choice">
                        <img src="{self.randomized_images[right_image]}" alt="Image {right_image}">
                        <div class="key-label">J</div>
                    </div>
                </div>
            """
    
    def add_feedback_trial(self, reward: bool, phase: str):
        """Add feedback trial after each choice"""
        self.time_elapsed += 1500
        
        if phase == "training":
            stimulus = '<div class="feedback reward">REWARD! +0.3 cents</div>' if reward else '<div class="feedback no-reward">No reward</div>'
        else:
            stimulus = '<div class="feedback no-feedback">response recorded</div>'
        
        feedback_trial = {
            "stimulus": stimulus,
            "response": None,
            "trial_type": "html-keyboard-response",
            "trial_index": len(self.data),
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        
        self.data.append(feedback_trial)
    
    def run_training_phase(self):
        """Run 80 training trials (8 blocks × 10 trials)"""
        trials_per_block = 10
        num_blocks = 8
        
        training_base_pairs = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]
        
        for block in range(1, num_blocks + 1):
            # Add block start indicator
            if block == 1:
                self._add_phase_indicator("training")
            
            # Get pairs for this block
            block_pairs = training_base_pairs + [(b, a) for a, b in training_base_pairs]
            random.shuffle(block_pairs)
            
            for trial_idx, pair in enumerate(block_pairs):
                trial_data = self.run_trial("training", block, pair, trial_idx)
                self.data.append(trial_data)
                self.add_feedback_trial(trial_data["reward"], "training")
            
            # Add break between blocks if needed
            if block == 4:
                self._add_break()
    
    def run_testing_phase(self):
        """Run 60 testing trials (2 blocks × 30 trials)"""
        trials_per_block = 30
        num_blocks = 2
        
        # Add testing phase indicator
        self._add_testing_transition()
        
        for block in range(1, num_blocks + 1):
            if block == 1:
                self._add_phase_indicator("testing")
            
            # For testing, we need all pairs twice
            block_pairs = self.all_pairs + [(b, a) for a, b in self.all_pairs]
            random.shuffle(block_pairs)
            
            for trial_idx, pair in enumerate(block_pairs):
                trial_data = self.run_trial("testing", block, pair, trial_idx)
                self.data.append(trial_data)
                self.add_feedback_trial(False, "testing")  # No actual reward in testing
    
    def _add_phase_indicator(self, phase: str):
        """Add phase start indicator"""
        self.time_elapsed += 2000
        text = "Training Block" if phase == "training" else "Testing Block"
        indicator = {
            "stimulus": f"<h3>{text}</h3>\n                <p>Press any key to start this block.</p>",
            "response": "f",
            "trial_type": "html-keyboard-response",
            "trial_index": len(self.data),
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(indicator)
    
    def _add_break(self):
        """Add break between blocks"""
        self.time_elapsed += 5000
        break_trial = {
            "stimulus": "<p>Take a break for a few seconds if needed.</p>\n                       <p>Press any key to continue</p>",
            "response": "f",
            "trial_type": "html-keyboard-response",
            "trial_index": len(self.data),
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(break_trial)
    
    def _add_testing_transition(self):
        """Add transition from training to testing"""
        self.time_elapsed += 5000
        transition = {
            "stimulus": "<h2>Training Complete!</h2>\n               <p>Now this is the testing phase, there will be no feedback, \n               but the underlying rule remains the same, and your bonus \n               will continue to accumulate.</p>\n               <p>Press any key to begin the testing phase.</p>",
            "response": "f",
            "trial_type": "html-keyboard-response",
            "trial_index": len(self.data),
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(transition)
    
    def add_initial_trials(self):
        """Add consent and instruction trials"""
        # Add consent
        self.time_elapsed = 15000
        consent = {
            "stimulus": "<h2><b>Consent Form</b></h2>...",  # Truncated for brevity
            "response": 0,
            "trial_type": "html-button-response",
            "trial_index": 0,
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(consent)
        
        # Add instructions
        self.time_elapsed += 30000
        instructions = {
            "view_history": [{"page_index": 0, "viewing_time": 30000}],
            "trial_type": "instructions",
            "trial_index": 1,
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(instructions)
        
        # Add randomization record
        self.time_elapsed += 10
        randomization = {
            "trial_type": "html-keyboard-response",
            "condition": self.condition,
            "randomized_images": self.randomized_images,
            "stimulus": "",
            "response": None,
            "trial_index": 2,
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(randomization)
        
        # Add ID collection
        self.time_elapsed += 4000
        id_collection = {
            "response": {"prolific_ID": self.subject_id},
            "trial_type": "survey-text",
            "trial_index": 3,
            "plugin_version": "2.1.0",
            "time_elapsed": self.time_elapsed
        }
        self.data.append(id_collection)
    
    def generate_data(self) -> List[Dict]:
        """Generate complete experimental data"""
        self._initialize_unchosen_counts()
        
        # Add initial trials
        self.add_initial_trials()
        
        # Run experiment phases
        self.run_training_phase()
        self.run_testing_phase()
        
        return self.data
    
    def save_data(self, filename: str = None):
        """Save data to JSON file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"random_agent_{self.subject_id}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        print(f"Data saved to {filename}")
        return filename

class PerfectAgent(RandomAgent):
    def __init__(self, subject_id: str = "perfect_agent_001", condition: str = "feature"):
        super().__init__(subject_id, condition)
        # Track unchosen counts for calculating probabilities
        self.perfect_unchosen_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
    
    def make_choice(self, left_image: str, right_image: str) -> Tuple[str, str, str]:
        """Perfect agent calculates reward probability for each option and chooses optimally"""
        
        # Calculate reward probability for choosing right (left becomes unchosen)
        n_right = self.perfect_unchosen_counts[right_image]
        prob_right = 1 - (0.7 ** (n_right + 1))
        
        # Calculate reward probability for choosing left (right becomes unchosen)
        n_left = self.perfect_unchosen_counts[left_image]
        prob_left = 1 - (0.7 ** (n_left + 1))
        
        # Choose the option with higher reward probability
        if prob_left > prob_right:
            chosen, unchosen, side = left_image, right_image, 'left'
        elif prob_right > prob_left:
            chosen, unchosen, side = right_image, left_image, 'right'
        else:
            # If equal probability, choose randomly
            if random.random() < 0.5:
                chosen, unchosen, side = left_image, right_image, 'left'
            else:
                chosen, unchosen, side = right_image, left_image, 'right'
        
        # Update perfect tracking (mimics what will happen in _calculate_reward)
        self.perfect_unchosen_counts[unchosen] += 1
        self.perfect_unchosen_counts[chosen] = 0
        
        return chosen, unchosen, side
    
    def _initialize_unchosen_counts(self):
        """Override to also reset perfect tracking"""
        super()._initialize_unchosen_counts()
        self.perfect_unchosen_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}

class QLearningAgent(RandomAgent):
    def __init__(self, subject_id: str = "qlearning_agent_001", condition: str = "feature",
                 learning_rate: float = 0.3, temperature: float = 1.0):
        super().__init__(subject_id, condition)
        self.learning_rate = learning_rate  # δ in the paper
        self.temperature = temperature      # T in the paper
        
        # Initialize Q-values for all images
        self.q_values = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 0.5, 'F': 0.5}
        
        # Track last choice for updating
        self.last_chosen = None
    
    def make_choice(self, left_image: str, right_image: str) -> Tuple[str, str, str]:
        """Choose based on softmax of Q-values"""
        
        # Get Q-values
        q_left = self.q_values[left_image]
        q_right = self.q_values[right_image]
        
        # Calculate softmax probabilities
        exp_left = np.exp(q_left / self.temperature)
        exp_right = np.exp(q_right / self.temperature)
        
        # Avoid division by zero
        if exp_left + exp_right > 0:
            prob_left = exp_left / (exp_left + exp_right)
        else:
            prob_left = 0.5
        
        # Make choice based on probability
        if random.random() < prob_left:
            chosen, unchosen, side = left_image, right_image, 'left'
        else:
            chosen, unchosen, side = right_image, left_image, 'right'
        
        # Store for later update
        self.last_chosen = chosen
        
        return chosen, unchosen, side
    
    def update_q_value(self, reward: bool):
        """Update Q-value of the last chosen image"""
        if self.last_chosen is None:
            return
        
        # Convert boolean reward to numeric (1 or 0)
        reward_value = 1.0 if reward else 0.0
        
        # Q-learning update rule: Q(s) ← Q(s) + δ[r - Q(s)]
        prediction_error = reward_value - self.q_values[self.last_chosen]
        self.q_values[self.last_chosen] += self.learning_rate * prediction_error
    
    def run_trial(self, phase: str, block: int, pair: Tuple[str, str], 
                  trial_in_block: int) -> Dict:
        """Override to include Q-value updates"""
        
        # Run parent trial
        trial_data = super().run_trial(phase, block, pair, trial_in_block)
        
        # Update Q-values based on reward
        self.update_q_value(trial_data["reward"])
        
        return trial_data
    
    def _initialize_unchosen_counts(self):
        """Override to also reset Q-values"""
        super()._initialize_unchosen_counts()
        self.q_values = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 0.5, 'F': 0.5}
        self.last_chosen = None

def find_best_qlearning_params():
    """Find best hyperparameters for Q-learning agent"""
    
    # Define parameter grid
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    temperatures = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    n_runs_per_combo = 100  # Multiple runs per parameter combination
    
    best_score = 0
    best_params = {}
    all_results = []
    
    print("Running grid search...")
    
    # Test all combinations
    for lr in learning_rates:
        print(f"Testing Learning Rate: {lr} ...")
        for temp in temperatures:
            print(f"Testing Temperature: {temp} ...")
            combo_rewards = []
            
            for run in range(n_runs_per_combo):
                agent = QLearningAgent(
                    subject_id="grid_test",
                    condition="feature",
                    learning_rate=lr,
                    temperature=temp
                )
                
                data = agent.generate_data()
                
                # Count rewards
                experimental_trials = [t for t in data if "phase" in t and t.get("phase") in ["training", "testing"]]
                rewards = sum(1 for t in experimental_trials if t.get("reward") == True)
                combo_rewards.append(rewards)
            
            avg_rewards = np.mean(combo_rewards)
            print(f"Average rewards for LR={lr} and T={temp}: {avg_rewards:.1f}/140 ({avg_rewards/140*100:.1f}%)")
            
            all_results.append({'lr': lr, 'temp': temp, 'score': avg_rewards})

            if avg_rewards > best_score:
                best_score = avg_rewards
                best_params = {'learning_rate': lr, 'temperature': temp}
                print(f"New best: LR={lr}, T={temp}, Avg rewards={avg_rewards:.1f}")
    
    print(f"\nOptimal parameters found:")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"Temperature: {best_params['temperature']}")
    print(f"Average rewards: {best_score:.1f}/140 ({best_score/140*100:.1f}%)")

    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    
    print("\n--- Top 5 Parameter Sets ---")
    for i in range(5):
        res = sorted_results[i]
        print(f"Rank {i+1}: LR={res['lr']}, T={res['temp']} -> Score: {res['score']:.2f}")
    
    return best_params

if __name__ == "__main__":
    
    output_folder = "agent_random"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    perfect_folder = "agent_perfect"
    if not os.path.exists(perfect_folder):
        os.makedirs(perfect_folder)
    
    ql_folder = "agent_Q"
    if not os.path.exists(ql_folder):
        os.makedirs(ql_folder)
    
    
    # Random Agent generation
    for i in range(1, 1001):
        # Create a random agent with unique ID
        agent = RandomAgent(subject_id=f"random_{i:03d}", condition="feature")
        
        # Generate data
        data = agent.generate_data()
        
        # Save to file with unique name
        filename = os.path.join(output_folder, f"random_agent_{i:03d}.json")
        agent.save_data(filename)
        
        # Print summary for each agent
        print(f"\nAgent {i:03d} Summary:")
    
        # Filter for actual experimental trials only (those with "phase" field)
        experimental_trials = [t for t in data if "phase" in t and t.get("phase") in ["training", "testing"]]
        training_trials = [t for t in experimental_trials if t.get("phase") == "training"]
        testing_trials = [t for t in experimental_trials if t.get("phase") == "testing"]

        # For rewards, count from both training and testing phases
        all_rewards = [t for t in experimental_trials if t.get("reward") == True]
    
        print("\nSummary:")
        print(f"Total trials: {len(experimental_trials)}")  
        print(f"Training trials: {len(training_trials)}")   
        print(f"Testing trials: {len(testing_trials)}")     
        print(f"Rewards earned: {len(all_rewards)}/{len(experimental_trials)} ({100*len(all_rewards)/len(experimental_trials):.1f}%)")
    
    
    # Perfect Agent generation
    for i in range(1, 1001):
        # Create a perfect agent with unique ID
        agent = PerfectAgent(subject_id=f"perfect_{i:03d}", condition="feature")
        
        # Generate data
        data = agent.generate_data()
        
        # Save to file with unique name
        filename = os.path.join(perfect_folder, f"perfect_agent_{i:03d}.json")
        agent.save_data(filename)
        
        # Print summary for each agent
        print(f"\nAgent {i:03d} Summary:")

        # Filter for actual experimental trials only (those with "phase" field)
        experimental_trials = [t for t in data if "phase" in t and t.get("phase") in ["training", "testing"]]
        training_trials = [t for t in experimental_trials if t.get("phase") == "training"]
        testing_trials = [t for t in experimental_trials if t.get("phase") == "testing"]

        # For rewards, count from both training and testing phases
        all_rewards = [t for t in experimental_trials if t.get("reward") == True]
    
        print("\nSummary:")
        print(f"Total trials: {len(experimental_trials)}")  
        print(f"Training trials: {len(training_trials)}")   
        print(f"Testing trials: {len(testing_trials)}")     
        print(f"Rewards earned: {len(all_rewards)}/{len(experimental_trials)} ({100*len(all_rewards)/len(experimental_trials):.1f}%)")
    

    # Q-Learning Agent generation
    best_params = find_best_qlearning_params()
    
    learning_rate = best_params['learning_rate']
    temperature = best_params['temperature']
    
    for i in range(1, 1001):
        
        agent = QLearningAgent(
            subject_id=f"q_{i:03d}",
            condition="feature",
            learning_rate=learning_rate,
            temperature=temperature
        )
        
        data = agent.generate_data()
        
        # Save with parameters in filename for reference
        filename = os.path.join(
            ql_folder, 
            f"q_agent_{i:03d}.json"
        )
        agent.save_data(filename)
        
        # Print summary
        experimental_trials = [t for t in data if "phase" in t and t.get("phase") in ["training", "testing"]]
        all_rewards = [t for t in experimental_trials if t.get("reward") == True]
        
        print(f"\nAgent {i:03d} (lr={learning_rate:.2f}, T={temperature:.2f}):")
        print(f"Rewards: {len(all_rewards)}/140 ({100*len(all_rewards)/140:.1f}%)")
        print(f"Final Q-values: {[f'{k}:{v:.2f}' for k,v in sorted(agent.q_values.items())]}")
        
